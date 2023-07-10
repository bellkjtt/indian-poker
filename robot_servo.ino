#include <SoftwareSerial.h>
#include <Servo.h> 

int blueTx=2;
int blueRx=3;
SoftwareSerial mySerial(blueTx, blueRx);

Servo servo;
int servoPin = 9;
int angle = 0; // servo position in degrees 
int servoDir, preServoDir = 0;
char a[20];
char c = 0; 

void setup() 
{ 
  servo.attach(servoPin);
  servo.write(50);
  mySerial.begin(9600);
  Serial.begin(9600);
} 
 
void loop() 
{
  if(mySerial.available()){
      while (mySerial.available()) {         //블루투스측 내용을 시리얼모니터에 출력
       a[c] = mySerial.read();
        c++;
      }
      int b = atoi(a);
      while (b > 0){
          servo.write(80);
          delay(1000);
          servo.write(50);
          delay(1000);
           b--;
           Serial.print(b);
            }
       delay(150);
       for(int i=0;i<21;i++) {
        a[i] = NULL;
      }
  c = 0;
  }
}
