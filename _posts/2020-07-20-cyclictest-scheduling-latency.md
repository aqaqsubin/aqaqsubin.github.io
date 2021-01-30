---
title: "KernelShark를 이용한 cyclictest Scheduling Latency 측정"
date: 2020-07-20 18:31:18 -0400
categories: Linux
---

이벤트를 처리하는 과정에서 설명되지 않는 지연은 RealTime Linux의 가장 큰 문제 중 하나다.  

RT Kernel의 context에서의 지연 시간(Latency)이란. 어떤 이벤트가 발생했을 때와 그 이벤트가 "handled" 되었을 때의 시간 간격을 의미한다.  

이를 측정하는 도구 중, `cyclictest` 라는 프로그램이 있는데, 이 프로그램을 통해 RT OS의 'RT'가 제대로 보장되고 있는지 확인할 수 있다.

cyclictest는 계속해서 타이머가 실행되는 task로 구성되어 있다.
이 때 타이머를 맞추고, 그 타이머가 직접 실행되는 시간까지의 간격을 event latency, 즉 scheduling latency로 측정할 수 있다.

<br>  

이제 kernelshark를 이용하여 커널의 scheduling latency를 직접 측정해보고, 
tracecmd 커맨드를 통해 측정한 latency가 맞는지 확인해보겠다.  

### **KernelShark 1.0 설치**

([https://kernelshark.org](https://kernelshark.org) : KernelShark 1.0 선택)

```
sudo apt-get install build-essential git cmake libjson-c-dev -y
sudo apt-get install freeglut3-dev libxmu-dev libxi-dev -y
sudo apt-get install qtbase5-dev -y
```
```
sudo apt-get install graphviz doxygen-gui -y
```
```
make install
make install_gui
```

KernelShark 1.0의 화면

![](/assets/images/scheduling_latency/kernelshark.png)

  
   <br>

### **Delta 계산**
  

![](/assets/images/scheduling_latency/graph_follows.png) *Graph follow 체크 >*![](/assets/images/scheduling_latency/markA.png) *Marker A 선택 > 원하는 이벤트 선택*

![](/assets/images/scheduling_latency/markA_evnet.png)

  

![](/assets/images/scheduling_latency/markB.png)
  *Marker B 선택 > 원하는 이벤트 선택*

![](/assets/images/scheduling_latency/markB_event.png)


Marker A와 Marker B의 Delta 값을 볼 수 있다. *(0.106 microsecond)*

![](/assets/images/scheduling_latency/markB_event.png)

   <br>

### **Event Filtering**  
  

왼쪽 상단의 *Filter > Advance Filtering*

![](/assets/images/scheduling_latency/filter.png)


   <br>

sched_wakeup 의 comm==’cyclictest’(pid = 17348)만을 통과하도록 필터 옵션 추가 *> Apply*
![](/assets/images/scheduling_latency/sched_wakeup_cyclictest.png)


  <br>

sched_switch 의 next_comm==’cyclictest’(pid = 17348)만을 통과하도록 필터 옵션 추가 *> Apply*
![](/assets/images/scheduling_latency/sched_switch_cyclictest.png)

  
cyclictest:17348 프로세스가 wakeup한 이후부터 sched_switch되기까지의 시간을 측정했다.  

![](/assets/images/scheduling_latency/latency_1.png)
  

![](/assets/images/scheduling_latency/latency_2.png)


![](/assets/images/scheduling_latency/latency_3.png)


![](/assets/images/scheduling_latency/latency_4.png)


![](/assets/images/scheduling_latency/latency_5.png)



<br>  


이제, 측정한 latency가 제대로 측정되었는지 trace-cmd로 확인해보자.  

### **Trace report**

sched_switch와 sched_wakeup 이벤트가 필터링되지 않은 상태에서
-w 옵션을 주어 report 를 진행하면 latency를 측정해준다.

![](/assets/images/scheduling_latency/tracecmd_command.png)


위 커맨드를 정리하면, 아래와 같다.  
-F 옵션으로 필터를 추가하고, -w 옵션을 통해 Latency의 평균과 최대 최소값을 측정한다.  
dat 파일은 읽을 수 없으므로, txt 파일로 복사해 저장한다.

```
trace-cmd -F [필터] -w [dat 파일 저장 경로] > [txt 파일 저장 경로]
```

KernelShark로 측정한 Latency와 이를 비교했을 때, 차이가 거의 없는 것을 확인할 수 있다.
![](/assets/images/scheduling_latency/tracecmd_report.png)  
  

-w을 추가하여 평균 latency, 최대 최소 latency를 확인 할 수 있다.
![](/assets/images/scheduling_latency/tracecmd_report_w.png)
