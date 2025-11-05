We use the official unitree mujoco repo to conduct the sim2sim verification in the Mujoco simulator.

Please make sure you clone the submodules and setup the unitree mujoco's dependencies according to the official readme : https://github.com/unitreerobotics/unitree_mujoco/tree/main?tab=readme-ov-file . You should only install the python sdk and cyclonedds. We strongly suggest you to install all these packages in the `thirdparties` directory.

After installing `unitree_sdk2_python`, please remember to modify the `thirdparties/unitree_sdk2_python/unitree_sdk2py/core/channel_config.py` file in case of permission errors:
```xml

Modify:

<OutputFile>/tmp/cdds.LOG</OutputFile>

---

Into:

<OutputFile>cdds.LOG</OutputFile>

```

