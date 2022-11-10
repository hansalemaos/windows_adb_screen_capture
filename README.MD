### Capture screen from background windows and adb (Windows only)

```python
pip install windows-adb-screen-capture
```

#### Update 2022/11/10

Faster ADB screenshots

#### Update 2022/11/04

- Screenshots can now be resized for imshow
- Screenshots can be taken from the whole screen, and even multiple screens 
- Bugfix: Closing and reopening imshow is working now

```python
from windows_adb_screen_capture import ScreenShots
sc2 = ScreenShots()
sc2.set_resize_ratio(50).choose_monitor_for_screenshot(0).imshow_screenshot_monitor() #0 for all screens, 1 for screen 1, 2 for screen 2...
```

#### Example with BlueStacks (using adb)

Open bluestacks 

```python
$adb connect localhost:5735  #connect to adb from shell or however you want 
from windows_adb_screen_capture import ScreenShots
sc=ScreenShots(hwnd=None, adb_path=r"C:\ProgramData\adb\adb.exe", adb_serial='localhost:5735')
sc.imshow_adb(sleep_time=0.05, quit_key="e") #show captured screen
```

<img src="https://github.com/hansalemaos/screenshots/raw/main/screencap1.png"/>

#### Edit screenshots before showing

```python
import cv2
sc.enable_show_edited_images() 

for x in range(120): #while True for endless loop
    tmpscreenshot = sc.imget_adb() #get as numpy array
    tmpscreenshort_inverted = cv2.bitwise_not(tmpscreenshot) #just an example, do your editing here
    sc.show_edited_image(tmpscreenshort_inverted) #show the edited pic

sc.disable_show_edited_images() #back to normal screen capturing
```

<img src="https://github.com/hansalemaos/screenshots/raw/main/screencap3.png"/>

#### Example with BlueStacks (using hwnd)

```python
sc2 = ScreenShots()
sc2.find_window_with_regex('[bB]lue[sS]tacks.*')
sc2.imshow_hwnd(sleep_time=0.05, quit_key="q")  #show captured screen
```

```python
#capture screen and edit before showing 
sc2.enable_show_edited_images()
for x in range(1000):   #while True for endless loop
    tmpscreenshot = sc2.imget_hwnd()
    tmpscreenshort_inverted = cv2.bitwise_not(tmpscreenshot) #do your editing here
    sc2.show_edited_image(tmpscreenshort_inverted) #show the edited pic
sc2.disable_show_edited_images()    #back to normal screen capturing
```

<img src="https://github.com/hansalemaos/screenshots/raw/main/screencap4.png"/>

#### Get screenshots without showing windows

```python
from windowcapture import ScreenShots
sc=ScreenShots(hwnd=None, adb_path=r"C:\ProgramData\adb\adb.exe", adb_serial='localhost:5735')
sc.imget_adb()
```

```python
sc2 = ScreenShots()
sc2.find_window_with_regex('[bB]lue[sS]tacks.*')
sc2.imget_hwnd()
```
