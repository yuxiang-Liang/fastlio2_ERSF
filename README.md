# Fast-LIO-ERSF

A robust LiDAR-inertial odometry (LIO) framework based on **Fast-LIO2**, featuring an **Extended Risk-Sensitive Filter (ERSF)** backend. This implementation is specifically designed to handle non-Gaussian noise and outliers in challenging industrial environments.

Unpublished work/Preprint. Shared for community feedback and reference. Hope it helps your research!

Related paper:
[Robust SLAM System Based on Extended Risk-Sensitive Filter](Robust%20SLAM%20System%20Based%20on%20Extended%20Risk-Sensitive%20Filter.pdf)

------

## Performance Evaluation

We conducted comparative experiments using our custom datasets. The metrics below represent the **Root Mean Square Error (RMSE)** of the Absolute Trajectory Error (ATE).

| Sequence      | Environment | FAST-LIO | **RS-FAST-LIO** | Point-LIO | **RS-Point-LIO** |
| ------------- | ----------- | -------- | --------------- | --------- | ---------------- |
| **Our_data1** | Cabin       | 0.0986   | **0.0951**      | 0.0908    | **0.0720**       |
| **Our_data2** | Pipeline    | 0.0945   | **0.0889**      | 0.1645    | **0.1349**       |



## Extensive Benchmark Results

We conducted an extensive evaluation using multiple public and experimental sequences. The following table shows the **Absolute Translational Error (RMSE, meters)**.

| Dataset       | Sequence     | FAST-LIO2 | **RS-FAST-LIO2** | Point-LIO | **RS-Point-LIO** |
| :------------ | :----------- | :-------- | :--------------- | :-------- | :--------------- |
| **Fastlio_1** | 100Hz        | 0.350     | **0.209**        | 0.339     | **0.108**        |
| **Fastlio_2** | Outdoor_Main | **0.051** | 0.100            | **0.014** | 0.022            |
| **Fastlio_3** | Outdoor_run  | 0.122     | 0.161            | 0.044     | **0.033**        |
| **R3live_1**  | Campus_00    | 0.077     | **0.068**        | 0.066     | **0.006**        |
| **R3live_2**  | Campus_02    | **0.019** | 0.031            | **0.037** | 0.048            |
| **R3live_3**  | Campus_03    | 0.089     | **0.039**        | **0.042** | 0.047            |
| **R3live_4**  | Hku_park_00  | **0.073** | 0.081            | **0.018** | 0.026            |
| **R3live_5**  | Hku_park_01  | 0.572     | **0.552**        | 0.557     | **0.490**        |
| **R3live_6**  | Hku_main_bld | 3.611     | **1.229**        | **0.061** | 1.617            |
| **R3live_7**  | Deg_seq_00   | 6.080     | **5.700**        | 5.531     | **4.884**        |
| **R3live_8**  | Deg_seq_02   | 17.626    | **17.574**       | 8.595     | **7.440**        |
| **Average**   |              | 3.160     | **2.793**        | 1.436     | **1.329**        |

## Acknowledgments

Thanks for [Fast-LIO2](https://github.com/hku-mars/FAST_LIO), Point-LIO.

## Reference

Detailed documentation and related papers can be found in the following directory: See the [reference](https://www.google.com/search?q=./reference) folder.

