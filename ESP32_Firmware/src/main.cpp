#include <Arduino.h>
#include <ESP32Servo.h>

Servo sorterServoRight;
Servo sorterServoLeft;

const int servoPinRight = 13; 
const int servoPinLeft = 32; 
const int trigPin = 26; 
const int echoPin = 25; 
const int thresholdDistance = 13; 

// Adjust this value to change speed. Higher = slower. (15ms is a good smooth sweep)
const int servoSpeedDelay = 15; 

// Helper function to read the sensor cleanly
int getDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 30000); 

  if (duration == 0) return 999; 
  return duration * 0.034 / 2;
}

// Helper function to sweep both servos simultaneously
void moveServosSlowly(int targetRight, int targetLeft) {
  // Read where the servos currently are
  int currentRight = sorterServoRight.read();
  int currentLeft = sorterServoLeft.read();

  // Loop until both servos reach their targets
  while (currentRight != targetRight || currentLeft != targetLeft) {
    
    // Step the right servo closer to its target
    if (currentRight < targetRight) currentRight++;
    else if (currentRight > targetRight) currentRight--;

    // Step the left servo closer to its target
    if (currentLeft < targetLeft) currentLeft++;
    else if (currentLeft > targetLeft) currentLeft--;

    // Update physical servos
    sorterServoRight.write(currentRight);
    sorterServoLeft.write(currentLeft);
    
    // Tiny pause dictates the speed of the sweep
    delay(servoSpeedDelay);
  }
}

void setup() {
  Serial.begin(115200); 

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  // 1. Attach, set to 90, wait for them to physically move, then detach to prevent jitter
  sorterServoRight.attach(servoPinRight);
  sorterServoLeft.attach(servoPinLeft);
  sorterServoRight.write(90); 
  sorterServoLeft.write(90); 
  delay(500); 
  sorterServoRight.detach();
  sorterServoLeft.detach();
}

void loop() {
  int distance = getDistance();

  // 2. Something in proximity is detected
  if (distance < thresholdDistance) {

    // --- NEW: SETTLE DELAY ---
    // Wait half a second for the hand/item to stop moving before snapping the photo
    delay(500); 

    Serial.println("SCAN"); 

    long startTime = millis();
    while (!Serial.available() && millis() - startTime < 5000) {
      delay(10); 
    }

    if (Serial.available()) {
      int targetAngle = Serial.parseInt(); 

      if (targetAngle == 0 || targetAngle == 180) {
        
        // --- 1-SECOND SAFETY DELAY ---
        // Give the user time to pull their hand away after the picture is snapped
        delay(1000); 
        
        // WAKE UP SERVOS: Reattach them so they can move
        sorterServoRight.attach(servoPinRight);
        sorterServoLeft.attach(servoPinLeft);
        
        // Ensure the library knows they are currently starting from 90
        sorterServoRight.write(90);
        sorterServoLeft.write(90);

        // Determine the target angles based on the model's prediction
        int rightAngle = (targetAngle == 0) ? 0 : 180;
        int leftAngle = (targetAngle == 0) ? 180 : 0;

        // 3. Both servos SWEEP to target angles
        moveServosSlowly(rightAngle, leftAngle);

        // 4. Wait until the object is removed OR 4-second timeout occurs (Failsafe)
        int currentDistance = getDistance();
        long waitStartTime = millis(); 
        
        while (currentDistance <= thresholdDistance + 2 && millis() - waitStartTime < 1000) {
          delay(100); 
          currentDistance = getDistance();
        }

        // 5. Object removed (or timeout hit)! Wait 700ms
        delay(700); 

        // 6. Both servos SNAP back to 90 instantly
        sorterServoRight.write(90); 
        sorterServoLeft.write(90);
        
        // Wait just long enough for the physical snap to finish (400ms)
        delay(400); 
        
        // PUT SERVOS TO SLEEP: Detach to kill the jitter
        sorterServoRight.detach();
        sorterServoLeft.detach();
      }
    }

    // Clear the serial buffer
    while(Serial.available()) Serial.read(); 
  }
  delay(100); 
}