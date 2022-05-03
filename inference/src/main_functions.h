//
// Created by matth on 4/23/2022.
//

#ifndef SIN_ESTIMATION_MAIN_H
#define SIN_ESTIMATION_MAIN_H


// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

// Initializes all data needed for the example. The name is important, and needs
// to be setup() for Arduino compatibility.
void setup();

// Runs one iteration of data gathering and inference. This should be called
// repeatedly from the application code. The name needs to be loop() for Arduino
// compatibility.
void loop();

#ifdef __cplusplus
}
#endif

#endif //SIN_ESTIMATION_MAIN_H
