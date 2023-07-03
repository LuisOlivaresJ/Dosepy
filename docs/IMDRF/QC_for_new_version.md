# Quality control tests for new versions or post installation acceptance

## Main window

Click on "ROI" buttton, make a ROI and click on "Corte" button.

    Images and profiles should to be correctly changed.
Open Dosepy/docs/Jupyter/**D_FILM.csv** as a reference distribution, writing 1 mm/point
Open Dosepy/docs/Jupyter/**D_TPS.csv** as distribution to be evaluated, writing 1 mm/point.

    Images and profiles should to be correctly displayed.
Do a calculation using a dose tolerance = 3, DTA = 2, dose threshold = 10. 
The output has to be 

    Dosis máxima: 8.5, 
    Umbral de dosis: 0.9, 
    Porcentaje de aprobación: 98.9%
    Índice gamma promedio: 0.3

Open Dosepy/docs/Jupyter/**D_20x20.csv** as a reference distribution, writing 0.78125 mm/point
Open Dosepy/docs/Jupyter/**RD_20x20cms_256x256pix.dcm** as distribution to be evaluated.

    Images and profiles should to be correctly displayed.

Click and move the cross hair.

    Profiles should to be correctly displayed.

## Film dosimetry window

Open Dosepy/docs/Jupyter/**Calibracion_Pre.tif**
Open Dosepy/docs/Jupyter/**Calibracion_Post.tif**

    Calibration curve should be displayed
    a0: 0.0097
    a1: 9.1473
    a2: 65.5534
    a3: 86.5541

Open Dosepy/docs/Jupyter/**QA_Pre.tif**
Open Dosepy/docs/Jupyter/**QA_Post.tif**

    An image should to be displayed.

Write 0.78125 mm/point and click "Reducir" button.

    A new images with 256 x 256 points should be displayed.

Check TIFF and CSV boxes, and "Guardar" button.

    As output, two new files are created.