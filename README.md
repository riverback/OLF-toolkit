code for OLF task

loss function is from https://github.com/JunMa11/SegLoss

Original Data
![CT data](https://github.com/riverback/OLF-toolkit/blob/main/presentation/raw_data_vis.png)

Annotation
![Annotation](https://github.com/riverback/OLF-toolkit/blob/main/presentation/nii_label_vis.png)

Color to Label

| Color    | Hexadecimal code | The thoracic spine | Label Value | RGB             |
| -------- | ---------------- | ------------------ | ----------- | --------------- |
| Red      | #e61948          | T1                 | 2.0         | (230, 25, 72)   |
| Orange   | #f58231          | T2                 | 4.0         | (245, 130, 49)  |
| Yellow   | #ffe119          | T3                 | 6.0         | (255, 255, 25)  |
| Green    | #3cb44b          | T4                 | 8.0         | (60, 180, 75)   |
| Cyan     | #42d4f4          | T5                 | 10.0        | (66, 212, 244)  |
| Blue     | #4363d8          | T6                 | 12.0        | (67, 99, 216)   |
| Magenta  | #f032e6          | T7                 | 14.0        | (240, 50, 230)  |
| Pink     | #fabed4          | T8                 | 16.0        | (250, 190, 212) |
| Beige    | #fffac8          | T9                 | 18.0        | (255, 250, 200) |
| Mint     | #aaffc3          | T10                | 20.0        | (170, 255, 195) |
| Lavender | #debeff          | T11                | 22.0        | (220, 190, 255) |
| Nacy     | #000075          | T12                | 24.0        | (0, 0, 117)     |
|          |                  |                    |             |                 |

![Color_to_label](https://github.com/riverback/OLF-toolkit/blob/main/presentation/color2label_vis.png)