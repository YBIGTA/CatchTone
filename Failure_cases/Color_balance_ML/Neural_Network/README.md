code based : https://github.com/davidsandberg/facenet<br>
Pre Traind data : [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)

Train datas from http://www.cs.sfu.ca/~colour/data/shi_gehler/<br>
learn_color_balance.py from https://github.com/opencv/opencv_contrib
****
Peter Gehler and Carsten Rother and Andrew Blake and Tom Minka and Toby Sharp, "Bayesian Color Constancy Revisited,"
Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2008. 
and http://www.kyb.mpg.de/bs/people/pgehler/colour/index.html.
****

### Architecture

Input --> [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) --> FC --> 3

3 --> R gamma, G gamma, B gamma ( It's idea from [that](https://docs.opencv.org/3.3.0/dc/dcb/tutorial_xphoto_training_white_balance.html) )

### Result


```
Iteration 396
cost:  1036328.1
out:  [-0.01459704  0.01489979  0.00497603]
ans:  [141. 314. 237.]
Iteration 495
cost:  1036333.5
out:  [-0.11118157  0.10429062  0.04645408]
ans:  [141. 314. 237.]
Iteration 594
cost:  1036351.0
out:  [ 0.02942188 -0.07662925  0.07567571]
ans:  [141. 314. 237.]
Iteration 693
cost:  1036227.2
out:  [0.07986249 0.03818512 0.0922645 ]
ans:  [141. 314. 237.]
Iteration 792
cost:  1036390.3
out:  [-0.03081712 -0.06545217 -0.02490767]
ans:  [141. 314. 237.]
Iteration 891
cost:  1036323.3
out:  [-0.00384281  0.00724577  0.04243239]
ans:  [141. 314. 237.]
Iteration 990
cost:  1036367.2
out:  [ 0.01934071 -0.06587644 -0.01239052]
ans:  [141. 314. 237.]
now image id :  26
gogogogoggogo
Iteration 99
cost:  915043.3
out:  [ 0.02020687 -0.04429521  0.08170161]
ans:  [ 694. 1458. 1503.]
Iteration 198
cost:  915142.9
out:  [ 0.01654812 -0.02341825 -0.06271861]
ans:  [ 694. 1458. 1503.]
Iteration 297
cost:  914979.4
out:  [ 0.07738215  0.07379781 -0.01325265]
ans:  [ 694. 1458. 1503.]
```


Cost never going down... <br>
I was 70 size of batch and 1000 iteration * 1000 number of re-learn(do not remove current learn stat..)
