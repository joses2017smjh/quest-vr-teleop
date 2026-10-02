#include <Servo.h>
#include <stdio.h>
#include <string.h>

// Outputs remain detached at boot. Commands use pulse widths, not assumed
// open/closed angles: P <pin:3|7|5|9> <microseconds:1000..2000>.
// D3 left claw, D7 right claw, D9 camera pan (left-right), D5 camera tilt (up-down).
// OFF releases all outputs; STATUS reports commanded values, not feedback.
const byte N = 4;
Servo claws[N];
const byte pins[N] = {3, 7, 5, 9};
// claws stay in 1000..2000; the camera SG90s use their full ~180 deg
const int minUs[N] = {1000, 1000, 500, 500};
const int maxUs[N] = {2000, 2000, 2500, 2500};
int currentUs[N] = {1500, 1500, 1500, 1500};
int targetUs[N] = {1500, 1500, 1500, 1500};
char line[48];
byte used = 0;
bool overflow = false;
unsigned long lastStep = 0;

void status() {
  for (byte i = 0; i < N; ++i) {
    Serial.print("D"); Serial.print(pins[i]);
    Serial.print(claws[i].attached() ? " ON " : " OFF ");
    Serial.print(currentUs[i]); Serial.print(" target=");
    Serial.println(targetUs[i]);
  }
}

void command() {
  if (!strcmp(line, "STATUS")) { status(); return; }
  if (!strcmp(line, "OFF")) {
    for (byte i = 0; i < N; ++i) claws[i].detach();
    Serial.println("OK OFF"); return;
  }
  int pin, pulse;
  char extra;
  byte i = N;
  if (sscanf(line, "P %d %d %c", &pin, &pulse, &extra) == 2)
    for (byte k = 0; k < N; ++k) if (pins[k] == pin) i = k;
  if (i == N || pulse < minUs[i] || pulse > maxUs[i]) {
    Serial.println("ERR use P 3|7 1000..2000, P 5|9 500..2500, STATUS, or OFF"); return;
  }
  if (!claws[i].attached()) {
    // Set the first pulse before attaching; physical position is unknown.
    currentUs[i] = pulse;
    claws[i].writeMicroseconds(pulse);
    claws[i].attach(pins[i], minUs[i], maxUs[i]);
  }
  targetUs[i] = pulse;
  Serial.println("OK");
  status();
}

void setup() {
  Serial.begin(115200);
  Serial.println("CLAW_D3_D7_CAM_D5_D9_READY outputs OFF");
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
  if (millis() - lastStep >= 20) {
    lastStep = millis();
    for (byte i = 0; i < N; ++i) {
      if (!claws[i].attached()) continue;
      int delta = targetUs[i] - currentUs[i];
      currentUs[i] += constrain(delta, -15, 15);   // 750 us/s
      claws[i].writeMicroseconds(currentUs[i]);
    }
  }
}
