
#include "Arduino.h"

#define SERVO_LEFT_PIN 6
#define SERVO_RIGHT_PIN 5
#define MOTOR_PIN 8
#define LED_PIN 4

void turn_servo(int d);
void controll_motor(int d);

void setup() {
    Serial.begin(9600);
    pinMode(SERVO_LEFT_PIN, OUTPUT);
    pinMode(SERVO_RIGHT_PIN, OUTPUT);
    pinMode(MOTOR_PIN, OUTPUT);
}

void loop() {
	Serial.setTimeout(100000);
	int deg = Serial.parseInt();
	if (deg != 0) {
		turn_servo(deg-1);
		controll_motor(deg);
		if (deg == 18) {
			digitalWrite(LED_PIN, HIGH);
		}
	}
}

void turn_servo(int d) {
	if (d <= 7 && d >= 0) {
		int val = (256.0/7.0)*(float)(7-d);
		if (val == 256) {
			analogWrite(SERVO_LEFT_PIN, val-1);
		} else {
			analogWrite(SERVO_LEFT_PIN, val);
		}
		analogWrite(SERVO_RIGHT_PIN, 0);
	} else if (d >= 8 && d < 15) {
		int val = (256.0/7.0)*(float)(d-7);
		if (val == 256) {
			analogWrite(SERVO_RIGHT_PIN, val-1);
		} else {
			analogWrite(SERVO_RIGHT_PIN, val);
		}
		analogWrite(SERVO_LEFT_PIN, 0);
	}
}

void controll_motor(int d) {
	if (d == 16) {
		digitalWrite(MOTOR_PIN, LOW);
	} else if (d == 17) {
		digitalWrite(MOTOR_PIN, HIGH);
	}
}