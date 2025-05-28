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
    A3 --> A4[Extract point features with PointNet]
    A4 --> A5[Generate Full Dataset 1000 Images]

    C[Train MLP] --> C1[Setup Backbone]
    C1 --> C2[Define Loss Function and Coding]
    C2 --> C3[Start Training]

    D[3D Set A] --> D1[Build with Depth Anything]
    D1 --> D2[Convert Depth Map to Point Cloud]

    B[3D Set B] --> B1[Connect COSMOS and VGGT]

    E[LLM Integration] --> E1[Find Language Parts in Code]
    E1 --> E2[Replace with LLM]

    F[MVP] --> F1[LLM Integration]
    F1 --> F2[3D Set A]
    F2 --> F3[coding API calls]
    F3 --> F4[evaluation]
```

- **Dataset**: coding api calls -> generate caption for small set (30) -> get 3D representation with VGGT -> generate whole dataset (1000)
- **3D setB**: connect cosmos and VGGT
- **Train MLP**: setup backbone -> setup loss function and coding -> start training
- **3D setA**: build depth anything -> depth map to point cloud
- **LLM**: find language parts in code -> replace with LLM
- **MVP**: TODO minimum viable product

### Important Dates

- 6/1 finish MVP
- 6/7 MLP training finish
- 6/10 poster presentation

### Milestones

### MVP

 
