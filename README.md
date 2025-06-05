# 3D-VIoLA: 3D Visual Information of Embodied Scene Views for Language-Action Prediction

All the shared information will be placed in this repo. Please frequently check for the updates and join the discussions.

### Pipeline

![](pipeline.png)

### Development

```mermaid
    flowchart TD
    A[Dataset] --> A1[Code API Calls]
    A1 --> A2[Generate Captions for 30 Images]
    A2 --> A3[Extract 3D Representation with VGGT]
    A3 --> A4[Generate Full Dataset 1000 Images]

    C[Train MLP] --> C1[Setup Backbone]
    C1 --> C2[Define Loss Function and Coding]
    C2 --> C3[Start Training]

    D[3D Set A] --> D1[Build with VGGT]

    B[3D Set B] --> B1[Connect COSMOS and VGGT]

    E[LLM Integration] --> E1[Find Language Parts in Code]
    E1 --> E2[Replace with LLM]

    F[MVP] --> F1[LLM Integration]
    F1 --> F2[3D Set A]
    F2 --> F3[coding API calls]
    F3 --> F4[evaluation]

    classDef completed fill:#c2f0c2,stroke:#2b8a3e,stroke-width:2px;
    classDef inprogress fill:#fff3cd,stroke:#ffcc00,stroke-width:2px,stroke-dasharray: 5 5;

    class A,A1,A2,A3 completed;
    class B completed;
    class C,C1,C2 completed;
    class D,D1 completed;
    class E,E1 completed;
    class A4 inprogress;
    class E2 inprogress;
    
```

### Important Dates

- 6/1 finish MVP
- 6/7 MLP training finish
- 6/10 poster presentation

### Milestones

- [ ] LLM integration
- [ ] Build Depth Anything
- [ ] Build PointNet++
- [ ] MVP
- [ ] Build dataset
- [ ] Train MLP

### MVP

Replace the MLP with API.

 
