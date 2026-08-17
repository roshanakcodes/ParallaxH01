#include <SPI.h>
#include <MFRC522.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <WiFi.h>
#include <HTTPClient.h>


// 1. Wi-Fi & Target Endpoint Configuration
const char* WIFI_SSID = "VITC-EVENT";
const char* WIFI_PASS = "H@ck17&18$";

// Local Laptop IP listening on port 5000
const char* SERVER_URL = "http://172.16.44.166:5000/api/scan";

// 2. Hardware Pin Assignments
#define RFID_SS_PIN   21
#define RFID_RST_PIN  22

#define TFT_CS        17
#define TFT_DC        16
#define TFT_RST        4

#define LED_BLUE      25
#define LED_RED       26
#define BUZZER_PIN    27

MFRC522 rfid(RFID_SS_PIN, RFID_RST_PIN);
Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);

// Audio & LED Indicators
void playVerifiedFeedback() {
  digitalWrite(LED_BLUE, HIGH);
  digitalWrite(LED_RED, LOW);
  tone(BUZZER_PIN, 3000, 150);
}

void playInvalidFeedback() {
  digitalWrite(LED_BLUE, LOW);
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_RED, HIGH);
    tone(BUZZER_PIN, 800);
    delay(150);
    digitalWrite(LED_RED, LOW);
    noTone(BUZZER_PIN);
    delay(150);
  }
}

// Display UI Renderers
void drawStandbyScreen() {
  pinMode(2, OUTPUT);
  digitalWrite(2, LOW); // Explicitly turns OFF built-in blue LED
  digitalWrite(LED_BLUE, LOW);
  digitalWrite(LED_RED, LOW);

  tft.fillScreen(ST77XX_BLACK);
  tft.fillRect(0, 0, 160, 24, ST77XX_BLUE);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(12, 8);
  tft.print("PARALLAX AI CHECKPOINT");

  tft.setTextColor(ST77XX_YELLOW);
  tft.setTextSize(1);
  tft.setCursor(10, 45);
  tft.print("SYSTEM STATUS:");

  tft.setTextColor(ST77XX_GREEN);
  tft.setTextSize(2);
  tft.setCursor(10, 60);
  tft.print("READY");

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(10, 95);
  tft.print(">> Tap Patient ID..");
}

void displayCapturedScreen(String uid, String targetPage) {
  tft.fillScreen(ST77XX_BLACK);
  tft.fillRect(0, 0, 160, 22, ST77XX_BLUE);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(15, 7);
  tft.print("CARD CAPTURED");

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(5, 35);
  tft.print("UID: ");
  tft.setTextColor(ST77XX_YELLOW);
  tft.println(uid);

  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(5, 55);
  tft.print("TARGET: ");
  tft.setTextColor(ST77XX_GREEN);
  tft.println(targetPage);

  playVerifiedFeedback();
}

void displayVerifiedScreen(String uid, String cause) {
  tft.fillScreen(ST77XX_BLACK);
  tft.fillRect(0, 0, 160, 22, ST77XX_GREEN);
  tft.setTextColor(ST77XX_BLACK);
  tft.setTextSize(1);
  tft.setCursor(15, 7);
  tft.print("VERIFIED");

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(5, 35);
  tft.print("UID: ");
  tft.setTextColor(ST77XX_YELLOW);
  tft.println(uid);

  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(5, 55);
  tft.print("STATUS: ");
  tft.setTextColor(ST77XX_GREEN);
  tft.println(cause);

  playVerifiedFeedback();
}

void displayRejectedScreen(String uid, String cause) {
  tft.fillScreen(ST77XX_BLACK);
  tft.fillRect(0, 0, 160, 22, ST77XX_RED);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(20, 7);
  tft.print("REJECTED");

  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(5, 35);
  tft.print("UID: ");
  tft.setTextColor(ST77XX_YELLOW);
  tft.println(uid);

  tft.setTextColor(ST77XX_RED);
  tft.setCursor(5, 55);
  tft.print("CAUSE:");
  tft.setCursor(5, 70);
  tft.print(cause);

  playInvalidFeedback();
}

// Network Request Handler
void processVerification(String uid, float weight, bool shake) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(SERVER_URL);
    http.setTimeout(20000);
    http.addHeader("Content-Type", "application/json");

    String payload = "{\"rfid_uid\":\"" + uid + "\",\"weight_g\":" + String(weight, 2) + ",\"shake_detected\":" + (shake ? "true" : "false") + "}";

    Serial.println("POSTing payload: " + payload);
    int httpCode = http.POST(payload);

    if (httpCode > 0) {
      String response = http.getString();
      Serial.println("Flask Response: " + response);

      // Flexible check tolerant of JSON spaces and formatting
      if (response.indexOf("CAPTURED") != -1) {
        displayCapturedScreen(uid, "FORM_PAGE");
      } 
      else if (response.indexOf("VERIFIED") != -1) {
        displayVerifiedScreen(uid, "ALL PASSED");
      } 
      else {
        // Extract cause code flexibly if rejected
        String cause = "REJECTED";
        int codeIdx = response.indexOf("\"code\"");
        if (codeIdx != -1) {
          int valStart = response.indexOf("\"", response.indexOf(":", codeIdx) + 1);
          int valEnd = response.indexOf("\"", valStart + 1);
          if (valStart != -1 && valEnd != -1) {
            cause = response.substring(valStart + 1, valEnd);
          }
        }
        displayRejectedScreen(uid, cause);
      }
    } else {
      Serial.printf("HTTP Error: %s\n", http.errorToString(httpCode).c_str());
      displayRejectedScreen(uid, "SERVER_OFFLINE");
    }
    http.end();
  } else {
    displayRejectedScreen(uid, "NO_WIFI");
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(LED_BLUE, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_BLUE, LOW);
  digitalWrite(LED_RED, LOW);

  pinMode(RFID_SS_PIN, OUTPUT);
  digitalWrite(RFID_SS_PIN, HIGH);
  pinMode(TFT_CS, OUTPUT);
  digitalWrite(TFT_CS, HIGH);

  SPI.begin();
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(1);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected! ESP32 IP: " + WiFi.localIP().toString());

  rfid.PCD_Init();
  drawStandbyScreen();
  Serial.println("--- Parallax System Ready ---");
}

void loop() {
  digitalWrite(TFT_CS, HIGH);

  if (!rfid.PICC_IsNewCardPresent()) return;
  if (!rfid.PICC_ReadCardSerial()) return;

  String cardUID = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    if (rfid.uid.uidByte[i] < 0x10) cardUID += "0";
    cardUID += String(rfid.uid.uidByte[i], HEX);
  }
  cardUID.toUpperCase();

  Serial.println("Scanned UID: " + cardUID);

  float measured_weight = 0.50; 
  bool shake_detected = false;

  processVerification(cardUID, measured_weight, shake_detected);

  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();

  delay(3000);
  drawStandbyScreen();
}