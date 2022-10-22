import base64
import subprocess

import win32gui, win32ui, win32con
import pandas as pd

from threading import Thread, Lock
from typing import Union
import numpy as np
import cv2
from time import sleep


def get_screenshot_adb(
    adb_executable=r"C:\ProgramData\adb\adb.exe",
    deviceserial="localhost:5735",
    brg_to_rgb=False,
):
    proc = subprocess.run(
        f"{adb_executable} -s {deviceserial} shell screencap -p | busybox base64",
        shell=False,
        capture_output=True,
    )
    png_screenshot_data = base64.b64decode(proc.stdout)
    images = cv2.imdecode(
        np.frombuffer(png_screenshot_data, np.uint8), cv2.IMREAD_COLOR
    )
    if brg_to_rgb:
        return bgr_to_rgb(images)
    return images


def bgr_to_rgb(src):
    if src.shape[1] == 3:
        dst = cv2.cvtColor(src, cv2.COLOR_BGR2BGR)
    else:
        dst = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
    return dst


class FeedEditedScreenshot:
    def __init__(self):
        self.enabled = False
        self.allimagestoshow = []

    def add_image(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        self._clearlist()
        self.allimagestoshow.append(img)

    def _clearlist(self):
        self.allimagestoshow = []


class WindowCapture:

    # constructor
    def __init__(self, hwnd):
        self.hwnd = hwnd
        self.window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = self.window_rect[2] - self.window_rect[0]
        self.h = self.window_rect[3] - self.window_rect[1]
        self.offset_x = self.window_rect[0]
        self.offset_y = self.window_rect[1]

    def get_window_position(self):
        self.window_rect = win32gui.GetWindowRect(self.hwnd)
        self.offset_x = self.window_rect[0]
        self.offset_y = self.window_rect[1]

    def get_screenshot(self, brg_to_rgb=False):
        self.get_window_position()
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (0, 0), win32con.SRCCOPY)
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype="uint8")
        img.shape = (self.h, self.w, 4)
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        if brg_to_rgb:
            img = bgr_to_rgb(src=img)
        img = np.ascontiguousarray(img.copy())

        return img


def get_all_hwnds_and_titles():
    dflist = []

    def winEnumHandler(hwnd, ctx):
        nonlocal dflist
        if win32gui.IsWindowVisible(hwnd):
            dflist.append((hwnd, hex(hwnd), win32gui.GetWindowText(hwnd)))

    win32gui.EnumWindows(winEnumHandler, None)
    return pd.DataFrame.from_records(
        dflist, columns=["aa_hwnd_int", "aa_hwnd_hex", "aa_title"]
    )


def find_window_with_regex(regular_expression):
    df = get_all_hwnds_and_titles()

    windowtitles = df.loc[
        df.aa_title.str.contains(regular_expression, regex=True, na=False)
    ]
    if not windowtitles.empty:
        return windowtitles["aa_hwnd_hex"].iloc[0]
    return None


class ScreenShots:
    def __init__(self, hwnd=None, adb_path=None, adb_serial="localhost:5555"):
        self.hwnd = hwnd
        self.windowsdf = pd.DataFrame()
        self.feed_edited_screenshot = FeedEditedScreenshot()
        self.adb_path = adb_path
        self.adb_serial = adb_serial
        self.show_edited_images = False
        self.adb_lock = Lock()
        self.hwnd_screenshot_grabber = None

    def enable_show_edited_images(self):
        self.show_edited_images = True
        self.feed_edited_screenshot.enabled = True
        return self

    def disable_show_edited_images(self):
        self.show_edited_images = False
        self.feed_edited_screenshot.enabled = False
        return self

    def find_window_with_regex(self, regular_expression):
        self.hwnd = int(find_window_with_regex(regular_expression), base=16)
        print(self.hwnd)
        return self

    def get_all_windows_with_handle(self):
        self.windowsdf = get_all_hwnds_and_titles()
        return self

    def imshow_adb(
        self,
        brg_to_rgb=False,
        sleep_time: Union[float, int, None] = 0.05,
        quit_key: str = "e",
    ):

        self.imshow_screenshot_adb(
            brg_to_rgb=brg_to_rgb,
            window_name=str(self.adb_serial),
            sleep_time=sleep_time,
            quit_key=quit_key,
        )

        return self

    def imget_adb(self, brg_to_rgb=False):
        return get_screenshot_adb(
            adb_executable=self.adb_path,
            deviceserial=self.adb_serial,
            brg_to_rgb=brg_to_rgb,
        )

    def imshow_hwnd(
        self,
        brg_to_rgb=False,
        sleep_time: Union[float, int, None] = 0.05,
        quit_key: str = "q",
    ):
        self.imshow_screenshot(
            window_name=str(self.hwnd),
            sleep_time=sleep_time,
            quit_key=quit_key,
            brg_to_rgb=brg_to_rgb,
        )
        return self

    def imget_hwnd(self, brg_to_rgb=False):
        if self.hwnd_screenshot_grabber is None:
            self.activate_hwnd_screenshot_grabber()
        return self.hwnd_screenshot_grabber.get_screenshot(brg_to_rgb=brg_to_rgb)

    def show_edited_image(self, numpyimg):
        self.adb_lock.acquire()
        self.feed_edited_screenshot.add_image(numpyimg)
        self.adb_lock.release()
        return self

    def imshow_screenshot_adb(
        self,
        brg_to_rgb=False,
        window_name: str = "",
        sleep_time: Union[float, int, None] = 0.05,
        quit_key: str = "q",
    ) -> None:

        t = Thread(
            target=self.cv_showscreenshotsadb,
            args=(brg_to_rgb, window_name, sleep_time, quit_key,),
        )
        t.start()

    def cv_showscreenshotsadb(
        self,
        brg_to_rgb=False,
        window_name: str = "",
        sleep_time: Union[float, int, None] = 0.05,
        quit_key: str = "q",
    ):
        while True:
            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                cv2.waitKey(0)
                # cv2.destroyAllWindows()
                return

            try:
                if self.feed_edited_screenshot.enabled and self.feed_edited_screenshot:
                    try:
                        cv2.imshow(
                            str(window_name),
                            self.feed_edited_screenshot.allimagestoshow[0],
                        )
                    except Exception:
                        pass

                else:
                    cv2.imshow(
                        str(window_name),
                        get_screenshot_adb(
                            adb_executable=self.adb_path,
                            deviceserial=self.adb_serial,
                            brg_to_rgb=brg_to_rgb,
                        ),
                    )
            except Exception as Fehler:
                print(Fehler)
                continue
            sleep(sleep_time)

    def imshow_screenshot(
        self,
        window_name: str = "",
        sleep_time: Union[float, int, None] = 0.05,
        quit_key: str = "q",
        brg_to_rgb=False,
    ) -> None:
        r"""
        """
        t = Thread(
            target=self.cv_showscreenshots,
            args=(window_name, sleep_time, quit_key, brg_to_rgb),
        )
        t.start()

    def activate_hwnd_screenshot_grabber(self):
        self.hwnd_screenshot_grabber = WindowCapture(self.hwnd)

    def cv_showscreenshots(
        self,
        window_name: str = "",
        sleep_time: Union[float, int, None] = 0.05,
        quit_key: str = "q",
        brg_to_rgb=False,
    ) -> None:
        if self.hwnd_screenshot_grabber is None:
            self.activate_hwnd_screenshot_grabber()
        while True:
            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                cv2.waitKey(0)
                return
            try:
                if self.feed_edited_screenshot.enabled and self.feed_edited_screenshot:
                    try:
                        cv2.imshow(
                            str(window_name),
                            self.feed_edited_screenshot.allimagestoshow[0],
                        )
                    except Exception:
                        pass

                else:
                    cv2.imshow(
                        str(window_name),
                        self.hwnd_screenshot_grabber.get_screenshot(
                            brg_to_rgb=brg_to_rgb
                        ),
                    )
            except Exception as Fehler:
                print(Fehler)
                continue
            sleep(sleep_time)
