#include <Arduino.h>
#include <string>

#include "main_functions.h"
#include "model.h"
#include "data.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


// Globals, used for compatibility with Arduino-style sketches.
namespace {
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;

    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    constexpr int kTensorArenaSize = 48 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];

    u_long average_latency;
}

// The name of this function is important for Arduino compatibility.
void setup() {
    // Inits RGB LED.
    pinMode(LEDR, OUTPUT);
    pinMode(LEDB, OUTPUT);
    pinMode(LEDG, OUTPUT);
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, LOW);

    //Inits serial I/O.
    Serial.begin(115200);

    digitalWrite(LEDB, LOW);

    delay(5000);

    // Inits error reporter.
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Loads model and checks compatibility.
    model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Model provided is schema version %d not equal "
                             "to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Loads all necessary arithmetic operations for densely connected layers.
    // These are the only layers we need for testing latency on basic feed-forward ANNs.
    static tflite::MicroMutableOpResolver<11> resolver;
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddFullyConnected();
    resolver.AddDequantize();

    // Inits interpreter, which effectively runs the model.
    static tflite::MicroInterpreter static_interpreter(model,
                                                       resolver,
                                                       tensor_arena,
                                                       kTensorArenaSize,
                                                       error_reporter);
    interpreter = &static_interpreter;

    // Allocates memory for model tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        digitalWrite(LEDR, LOW);
        digitalWrite(LEDG, HIGH);
        digitalWrite(LEDB, HIGH);

        TF_LITE_REPORT_ERROR(error_reporter, "Tensor memory allocation failed");

        return;
    } else {
        Serial.println("Tensor memory allocation OK");
    }

    // Declares pointers to model input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Copies the dummy 1D array input data into the input tensor, byte by byte.
    size_t current_byte = 0;
    for (size_t sample; sample < NUM_PDS * NUM_SAMPLES; sample++) {
        input->data.f[current_byte++] = DUMMY_DATA[sample];
    }

    // Invokes the model on the dummy data once to check for errors.
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        digitalWrite(LEDR, LOW);
        digitalWrite(LEDG, HIGH);
        digitalWrite(LEDB, HIGH);

        TF_LITE_REPORT_ERROR(error_reporter, "Model invocation failed");

        return;
    } else {
        Serial.println("Model invocation OK");
    }

    Serial.println("Starting latency test...");

    digitalWrite(LEDG, LOW);
    digitalWrite(LEDB, HIGH);

    // Finds the average time it takes to invoke the model across 100 invocations.
    const int repetitions = 100;

    u_long start = micros();
    for (int i = 0; i < repetitions; i++) {
        interpreter->Invoke();
    }
    u_long end = micros();

    average_latency = (end - start) / repetitions;

    digitalWrite(LEDG, HIGH);
}

// The name of this function is important for Arduino compatibility.
void loop() {
    Serial.println("Average Invocation Latency: " + String(average_latency));
    delay(1000);
}