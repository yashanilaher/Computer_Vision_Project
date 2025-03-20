# Project Architecture

The **Gesture Control Game** project is designed with a modular architecture to ensure scalability and maintainability. Below is an overview of the system architecture and components.

## System Workflow

The following diagram illustrates the flow of data from the webcam input to the final game control:

```mermaid
graph TD;
    A[Webcam Input] -->|Frame Capture| B[Gesture Recognition];
    B -->|Body Posture| C[Body Gesture Detection];
    B -->|Hand Gesture| D[Hand Gesture Detection];
    C -->|Move Left/Right| E[Game Control];
    C -->|Jump/Crouch| E;
    D -->|Move Left/Right| E;
    D -->|Jump/Crouch| E;
    E -->|Game Actions| F[Game];
```