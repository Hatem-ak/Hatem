int button = A13;
int RedLed = A0; 
int RedLed1 = A1;
int RedLed2 = A2;
int RedLed3 = A3;
int GreenLed = A6;
int GreenLed1 = A7;
int GreenLed2 = A8;
int GreenLed3 = A9;
int YellowLed = A15;
int i;
int data;
void setup() {
  Serial.begin(11520);
  pinMode(button, INPUT);
  pinMode(RedLed, OUTPUT);  
  pinMode(RedLed1, OUTPUT);
  pinMode(RedLed2, OUTPUT);
  pinMode(RedLed3, OUTPUT);
  pinMode(GreenLed, OUTPUT);  
  pinMode(GreenLed1, OUTPUT);
  pinMode(GreenLed2, OUTPUT);
  pinMode(GreenLed3, OUTPUT);
  pinMode(YellowLed, OUTPUT);
}
 
void loop() {
  
  if (digitalRead(button) == 1)
  {digitalWrite(YellowLed, HIGH);}
  else{
  digitalWrite(YellowLed,LOW); 
  }
  if(Serial.available() > 0) {
  data = Serial.read();
  if (data == '1'){
  digitalWrite(RedLed, HIGH); 
  digitalWrite(RedLed1, HIGH);
  digitalWrite(RedLed2, HIGH);
  digitalWrite(RedLed3, HIGH);}

  if (data == '2'){
  digitalWrite(GreenLed, HIGH); 
  digitalWrite(GreenLed1, HIGH);
  digitalWrite(GreenLed2, HIGH);
  digitalWrite(GreenLed3, HIGH);}
  }
  
 
  }
