# -School-of-Future-Technology.BIT-
本源码是北京理工大学未来精工技术2022级“智能无人系统总体技术”课程结课项目中“空地协同”任务的源码。基于ClaraUE4仿真平台、kyxz虚拟环境和python完成任务。引入两个无人车、两架无人机完成空地协同。涉及A*算法、骨架细化算法、PID控制、Stanley控制等算法与控制方法，同时加入动态增益系数与预测技术提升了仿真的稳定性。

源码中含有main.py与controller.py。其中controller.py需在lib文件夹中替换原controller.py文件，main.py文件在lib文件夹上一级即可。

使用步骤：

1.打开ClaraUE4仿真平台。

2.打开simu_drever_windows中的start.bat文件，并选择选项2.四旋翼无人机。

3.打开subject4中的subject4.exe文件。

4.在已配置好kyxz环境的VScode中运行main.py文件。

我们希望后届同学在使用本源码的时候，尽量对源码所涉及的算法、方法进行更多优化，以达到更好的效果。切记不要照抄代码不作任何更改。代码运行效果与电脑的性能强关联，源码在不同电脑上运行的稳定性、优越性可能不同。

本源码仅供北京理工大学未来精工技术专业内部参考学习使用，禁止用于任何商业用途。
