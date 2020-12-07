# python-iir
DSP FFT noise reduction

## Environment
Python3.6.7  
scipy1.5.4  
matplotlib3.3.2  
numpy 1.19.2  



## 这是什么
哈尔滨工业大学数字信号处理课程（彭宇老师）布置的实验作业。  
在Python3.6.7，scipy1.5.4环境下运行正常。  


## 怎么用/How to use
./data里面是要处理的4个mp3以及用./data/mp3ToWav.py转换成的四个wav文件。  
如果要处理《我和你》：  
```python
python final.py
```
  
如果要处理《广播电台》：  
```python
python final2.py
```
如果要将MP3文件转换成wav文件，将要转换的MP3放入./data文件夹中，并且运行  
```python
pip install pydub  
python ./data/mp3ToWav.py  
```

每个代码里有注释。  


## 注意
1. iirLowPass()函数的输入wave是原始信号波形，不是FFT变换之后的频谱图，scipy.signal.lfilter()和scipy.signal.filtfilt()同理  
2. 《广播电台》的加噪前后两个wav文件的采样频率不同，这两个wav的质量不一样，才会导致画频谱图时横坐标范围不一样。（这是老师的问题，不是我的）（我笑一下）  
3. 《广播电台》降噪效果不好是因为实验指导书上要求一个低通滤波器，但实验发现噪声的频率很低，需要一个带阻滤波器去降噪，有时间可以自己试验一下，我大概率不会再做了，很简单把iirLowPass()函数改一下就行。  


## 还有一件事
注释应该写的还可以吧，看不懂的话也不用管了。  
