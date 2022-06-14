# Eye Region Labeling Tool

We open eye key-point labeling tool source consists of total 50 points


Also provide part of MPIIGAZE labled by us

## **License**   
The Datasets are under the license of [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/)


## **Installation**    
```python
pip install -r requirements.txt
```


## **Demo with example** 

### 1. Run the script below

```python
python labeling_tool_line.py 
```

<img src="https://user-images.githubusercontent.com/67889349/162893324-66e060d9-a27d-40c4-853f-705ce8b3dd38.png" widht="300" height="200">

<br/>

### 2. Dot the each end-points in eye width ( 2 points )
<img src="https://user-images.githubusercontent.com/67889349/162893892-fe69c329-b68a-4380-a192-4afe1bef9117.png" widht="300" height="200">

<br/>

### 3. Dot the each points where green lines overlap eye edge ( 14 points )
we provide a little correction to line to put points on lines

<img src="https://user-images.githubusercontent.com/67889349/162894506-28c92b88-05ec-4e88-9ff1-146247a5106e.png" widht="300" height="200"> 

<br/>

### 4. Dot the each points with the edge of the iris ( 8 points )
If you take 8 points, 


<img src="https://user-images.githubusercontent.com/67889349/162895639-33c19d67-e12d-40d2-8bf6-7a0a4a3d99da.png" widht="300" height="200"> 

<br/>

It finds the ellipse that suits you best and get 32 points. Each coordinate of points is saved in csv format

<img src="https://user-images.githubusercontent.com/67889349/162896787-5f89d6ee-22d2-4d79-bc6b-74a5a3856747.png" widht="300" height="200"> 




 