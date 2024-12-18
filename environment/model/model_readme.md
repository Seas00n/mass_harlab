Name: MyoSuite
Version: 2.8.0

路径说明：
myosuite模型路径在
/.local/lib/python3.8/site-packages/myosuite/envs/myo/assets/leg/
可通过mujoco simulate拖动查看
cd /.local/lib/python3.8/site-packages/myosuite/envs/myo/assets/leg/
./simulate

该模型为混合模型，
上半身Rajagopal2015_converted.xml通过关节运动，在Raj文件夹内
下半身为自带模型myolegs_osl_assets.xml通过肌肉带动
上半身所需要的stl文件在raj中

将Raja和myoosl_runtrack2.xml放在myoosl_runtrack下
将raj放在/myosuite/simhive/myo_sim/下
