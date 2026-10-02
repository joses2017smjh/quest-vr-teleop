#include <Servo.h>
#include <stdio.h>
#include <string.h>

// Claw driver with runtime pin discovery.
//
// The servos are not on D3/D5, so the pin assignment is a command rather than a
// constant. Servo drives its outputs from a Timer1 ISR, so any digital pin
// works and the PWM marking on the silkscreen does not matter. D0/D1 are
// refused because they are the serial link back to the host.
//
//   SCAN [from] [to]  wiggle each pin in turn so you can see which one moves;
//                     any serial input aborts and detaches every output
//   P <pin> <us>      command a pin, 1000..2000, slewed SLEW_US per 20ms
//   OFF [pin]         detach one pin, or all of them
//   STATUS            commanded values; there is no position feedback
//
// Run SCAN with the claws mechanically free. Outputs stay detached until
// commanded, so the pins float at boot exactly as they do with no firmware.

const byte MAX_SLOTS = 4;
const byte FIRST_PIN = 2;    // D0/D1 are the serial link
const byte LAST_PIN = 19;    // A0..A5 are 14..19 and usable as digital
const int CENTER_US = 1500;
const int MIN_US = 1000;
const int MAX_US = 2000;
const int SLEW_US = 5;       // per 20ms tick, same rate as the D3/D5 sketch
const int SCAN_SWING = 120;  // deliberately small: identify, do not travel

Servo servos[MAX_SLOTS];
byte slotPin[MAX_SLOTS];
int currentUs[MAX_SLOTS];
int targetUs[MAX_SLOTS];
bool active[MAX_SLOTS];

char line[48];
byte used = 0;
bool overflow = false;
unsigned long lastStep = 0;

int slotOf(byte pin) {
  for (byte i = 0; i < MAX_SLOTS; ++i)
    if (active[i] && slotPin[i] == pin) return i;
  return -1;
}

int claim(byte pin, int us) {
  int i = slotOf(pin);
  if (i >= 0) return i;
  for (i = 0; i < MAX_SLOTS; ++i) {
    if (active[i]) continue;
    slotPin[i] = pin;
    currentUs[i] = us;
    targetUs[i] = us;
    active[i] = true;
    // Set the first pulse before attaching; physical position is unknown.
    servos[i].writeMicroseconds(us);
    servos[i].attach(pin, MIN_US, MAX_US);
    return i;
  }
  return -1;
}

void release(byte i) {
  servos[i].detach();
  active[i] = false;
}

void releaseAll() {
  for (byte i = 0; i < MAX_SLOTS; ++i)
    if (active[i]) release(i);
}

void tick() {
  if (millis() - lastStep < 20) return;
  lastStep = millis();
  for (byte i = 0; i < MAX_SLOTS; ++i) {
    if (!active[i]) continue;
    int delta = targetUs[i] - currentUs[i];
    currentUs[i] += constrain(delta, -SLEW_US, SLEW_US);
    servos[i].writeMicroseconds(currentUs[i]);
  }
}

void status() {
  bool any = false;
  for (byte i = 0; i < MAX_SLOTS; ++i) {
    if (!active[i]) continue;
    any = true;
    Serial.print("D"); Serial.print(slotPin[i]);
    Serial.print(" ON "); Serial.print(currentUs[i]);
    Serial.print(" target="); Serial.println(targetUs[i]);
  }
  if (!any) Serial.println("none attached, all pins floating");
}

// Slew one slot to us and hold for ms. False means the operator asked to stop.
bool settle(byte i, int us, unsigned int ms) {
  targetUs[i] = us;
  unsigned long end = millis() + ms;
  while ((long)(millis() - end) < 0) {
    if (Serial.available()) return false;
    tick();
  }
  return true;
}

void scan(byte from, byte to) {
  releaseAll();
  Serial.println("SCAN start - send any character to abort");
  for (byte pin = from; pin <= to; ++pin) {
    int i = claim(pin, CENTER_US);
    if (i < 0) { Serial.println("ERR no free slot"); break; }
    Serial.print("  testing D"); Serial.println(pin);
    bool ok = settle(i, CENTER_US, 400) &&
              settle(i, CENTER_US + SCAN_SWING, 700) &&
              settle(i, CENTER_US - SCAN_SWING, 700) &&
              settle(i, CENTER_US, 500);
    release(i);
    if (!ok) {
      while (Serial.available()) Serial.read();
      Serial.println("SCAN aborted, all outputs off");
      return;
    }
  }
  Serial.println("SCAN done, all outputs off");
}

void command() {
  int a, b;
  char extra;

  if (!strcmp(line, "STATUS")) { status(); return; }
  if (!strcmp(line, "OFF")) { releaseAll(); Serial.println("OK OFF all"); return; }
  if (!strcmp(line, "SCAN")) { scan(FIRST_PIN, LAST_PIN); return; }

  if (sscanf(line, "SCAN %d %d %c", &a, &b, &extra) == 2) {
    if (a < FIRST_PIN || b > LAST_PIN || a > b) { Serial.println("ERR bad range"); return; }
    scan(a, b);
    return;
  }
  if (sscanf(line, "OFF %d %c", &a, &extra) == 1) {
    int i = slotOf(a);
    if (i < 0) { Serial.println("ERR that pin is not attached"); return; }
    release(i);
    Serial.println("OK OFF");
    return;
  }
  if (sscanf(line, "P %d %d %c", &a, &b, &extra) == 2) {
    if (a < FIRST_PIN || a > LAST_PIN) { Serial.println("ERR pin must be 2..19"); return; }
    if (b < MIN_US || b > MAX_US) { Serial.println("ERR us must be 1000..2000"); return; }
    int i = claim(a, b);
    if (i < 0) { Serial.println("ERR no free slot, OFF something first"); return; }
    targetUs[i] = b;
    Serial.println("OK");
    status();
    return;
  }
  Serial.println("ERR use SCAN [from to], P <pin> <us>, OFF [pin], STATUS");
}

void setup() {
  Serial.begin(115200);
  Serial.println("CLAW_FINDER_READY outputs OFF");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[used] = 0;
      if (overflow) Serial.println("ERR line too long");
      else if (used) command();
      used = 0; overflow = false;
    } else if (used < sizeof(line) - 1) line[used++] = c;
    else overflow = true;
  }
  tick();
}
