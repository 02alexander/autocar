
#include "Arduino.h"

#define SERVO_LEFT_PIN 6
#define SERVO_RIGHT_PIN 5
#define MOTOR_PIN 8
#define LED_PIN 4

void turn_servo(int d);
void controll_motor(int d);

bool led_is_on = false;

void setup() {
    Serial.begin(9600);
    pinMode(SERVO_LEFT_PIN, OUTPUT);
    pinMode(SERVO_RIGHT_PIN, OUTPUT);
    pinMode(MOTOR_PIN, OUTPUT);
}

void loop() {
	if (Serial.available()) {
		int deg = Serial.parseInt();
		if (deg != 0) {
			if (!led_is_on) {
				digitalWrite(LED_PIN, HIGH);
				led_is_on = true;
			}
			if (deg==20) {
				analogWrite(SERVO_LEFT_PIN, 0);
				analogWrite(SERVO_RIGHT_PIN, 0);
			}
			turn_servo(deg-1);
			controll_motor(deg);
		}
	}
}

void turn_servo(int d) {
	if (d <= 7 && d >= 0) {
		int val = (256.0/7.0)*(float)(7-d)+10;
		if (val >= 256) {
			analogWrite(SERVO_LEFT_PIN, 255);
		} else {
			analogWrite(SERVO_LEFT_PIN, val);
		}
		analogWrite(SERVO_RIGHT_PIN, 0);
	} else if (d >= 8 && d < 15) {
		int val = (256.0/7.0)*(float)(d-7)+10;
		if (val >= 256) {
			analogWrite(SERVO_RIGHT_PIN, 255);
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