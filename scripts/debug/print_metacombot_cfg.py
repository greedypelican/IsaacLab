from isaaclab.app import AppLauncher

# 1) 앱 런처 생성 (headless면 시각화 비활성)
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# 2) 이후에 isaaclab 관련 모듈 import
from isaaclab_assets.robots.metacombotx import METACOMBOTX_CFG

print("USD path :", METACOMBOTX_CFG.spawn.usd_path)
print("Root prim:", METACOMBOTX_CFG.prim_path)

# 3) 종료
simulation_app.close()
