"""文件列表点击式多选（点一次选中、再点取消）行为测试。"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from uvr_lite.ui.main import ToggleSelectList


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_list(names):
    lst = ToggleSelectList()
    lst.addItems(names)
    lst.resize(300, 200)  # 保证所有行可见、visualItemRect 有效
    lst.show()
    return lst


def _click(qapp, lst, row, wait=300):
    rect = lst.visualItemRect(lst.item(row))
    QTest.mouseClick(lst.viewport(), Qt.LeftButton, pos=rect.center())
    qapp.processEvents()
    if wait:
        QTest.qWait(wait)  # 避开 Qt 双击合并（双击默认会改为单选）


def test_click_selects_then_deselects(qapp):
    lst = _make_list(["a.wav", "b.wav"])
    qapp.processEvents()

    _click(qapp, lst, 0)
    assert lst.item(0).isSelected(), "第一次点击应选中"
    _click(qapp, lst, 0)
    assert not lst.item(0).isSelected(), "再点同一项应取消选中"


def test_click_two_items_selects_both(qapp):
    lst = _make_list(["a.wav", "b.wav"])
    qapp.processEvents()

    _click(qapp, lst, 0)
    _click(qapp, lst, 1)
    assert lst.item(0).isSelected() and lst.item(1).isSelected(), "多选互不影响"


def test_click_blank_keeps_selection(qapp):
    """点击空白区域不清空已选中的项（多选操作更安全）。"""
    lst = _make_list(["a.wav"])
    qapp.processEvents()
    _click(qapp, lst, 0)
    assert lst.item(0).isSelected()

    blank = lst.viewport().rect().bottomLeft()
    blank += __import__("PySide6.QtCore", fromlist=["QPoint"]).QPoint(5, -1)
    QTest.mouseClick(lst.viewport(), Qt.LeftButton, pos=blank)
    qapp.processEvents()
    assert lst.item(0).isSelected(), "点击空白不应误清选择"
