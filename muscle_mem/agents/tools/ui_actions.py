from typing import Any, Dict, List, Optional

from muscle_mem.agents.tools.registry import tool_action


UBUNTU_APP_SETUP = f"""import subprocess;
import difflib;
import pyautogui;
pyautogui.press('escape');
time.sleep(0.5);
output = subprocess.check_output(['wmctrl', '-lx']);
output = output.decode('utf-8').splitlines();
window_titles = [line.split(None, 4)[2] for line in output];
closest_matches = difflib.get_close_matches('APP_NAME', window_titles, n=1, cutoff=0.1);
if closest_matches:
    closest_match = closest_matches[0];
    for line in output:
        if closest_match in line:
            window_id = line.split()[0]
            break;
subprocess.run(['wmctrl', '-ia', window_id])
subprocess.run(['wmctrl', '-ir', window_id, '-b', 'add,maximized_vert,maximized_horz'])
"""


SET_CELL_VALUES_CMD = """import uno
import subprocess

def identify_document_type(component):
    if component.supportsService("com.sun.star.sheet.SpreadsheetDocument"):
        return "Calc"

    if component.supportsService("com.sun.star.text.TextDocument"):
        return "Writer"

    if component.supportsService("com.sun.star.sheet.PresentationDocument"):
        return "Impress"

    return None

def cell_ref_to_indices(cell_ref):
    column_letters = ''.join(filter(str.isalpha, cell_ref))
    row_number = ''.join(filter(str.isdigit, cell_ref))

    col = sum((ord(char.upper()) - ord('A') + 1) * (26**idx) for idx, char in enumerate(reversed(column_letters))) - 1
    row = int(row_number) - 1
    return col, row

def set_cell_values(new_cell_values: dict[str, str], app_name: str = "Untitled 1", sheet_name: str = "Sheet1"):
    new_cell_values_idx = {{}}
    for k, v in new_cell_values.items():
        try:
            col, row = cell_ref_to_indices(k)
        except:
            col = row = None

        if col is not None and row is not None:
            new_cell_values_idx[(col, row)] = v

    # Clean up previous TCP connections.
    subprocess.run(
        'echo \"password\" | sudo -S ss --kill --tcp state TIME-WAIT sport = :2002',
        shell=True,
        check=True,
        text=True,
        capture_output=True
    )

    # Dynamically allow soffice to listen on port 2002.
    subprocess.run(
        [
            "soffice",
            "--accept=socket,host=localhost,port=2002;urp;StarOffice.Service"
        ]
    )

    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context
    )
    context = resolver.resolve(
        f"uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
    )
    desktop = context.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", context
    )

    # Collect all LibreOffice-related opened windows.
    documents = []
    for i, component in enumerate(desktop.Components):
        title = component.Title
        doc_type = identify_document_type(component)
        documents.append((i, component, title, doc_type))

    # Find the LibreOffice Calc app and the sheet of interest.
    spreadsheet = [doc for doc in documents if doc[3] == "Calc"]
    selected_spreadsheet = [doc for doc in spreadsheet if doc[2] == app_name]
    if spreadsheet:
        try:
            if selected_spreadsheet:
                spreadsheet = selected_spreadsheet[0][1]
            else:
                spreadsheet = spreadsheet[0][1]

            sheet = spreadsheet.Sheets.getByName(sheet_name)
        except:
            raise ValueError(f"Could not find sheet {{sheet_name}} in {{app_name}}.")

        for (col, row), value in new_cell_values_idx.items():
            cell = sheet.getCellByPosition(col, row)

            # Set the cell value.
            if isinstance(value, (int, float)):
                cell.Value = value
            elif isinstance(value, str):
                if value.startswith("="):
                    cell.Formula = value
                else:
                    cell.String = value
            elif isinstance(value, bool):
                cell.Value = 1 if value else 0
            elif value is None:
                cell.clearContents(0)
            else:
                raise ValueError(f"Unsupported cell value type: {{type(value)}}")

    else:
        raise ValueError(f"Could not find LibreOffice Calc app corresponding to {{app_name}}.")

set_cell_values(new_cell_values={cell_values}, app_name="{app_name}", sheet_name="{sheet_name}")        
"""


class UIActions:
    def __init__(self, grounding) -> None:
        self.grounding = grounding

    @tool_action
    def click(
        self,
        element_description: str,
        num_clicks: int = 1,
        button_type: str = "left",
        hold_keys: List = [],
    ):
        """点击元素。
        IMPORTANT: 本方法用于GUI操作。如果要点击图片或者图片中的某个元素，请使用`click_image`方法。

        Args:
            element_description:str
            num_clicks:int, number of times to click the element
            button_type:str, which mouse button to press can be "left", "middle", or "right"
            hold_keys:List, list of keys to hold while clicking
        """

        coords1 = self.grounding.generate_coords(
            element_description, self.grounding.obs
        )
        x, y = self.grounding.resize_coordinates(coords1)
        command = "import pyautogui; "

        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"import pyautogui; pyautogui.click({x}, {y}, clicks={num_clicks}, button={repr(button_type)}); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
        return command

    @tool_action
    def click_image(
        self,
        element_description: str,
        num_clicks: int = 1,
        button_type: str = "left",
        hold_keys: List = [],
    ):
        """点击某个图片，或者图片上的某个元素"""

        coords1 = self.grounding.generate_coords(
            element_description, self.grounding.obs, use_image_model=True
        )
        x, y = self.grounding.resize_coordinates(coords1)
        command = "import pyautogui; "

        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"import pyautogui; pyautogui.click({x}, {y}, clicks={num_clicks}, button={repr(button_type)}); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
        return command

    @tool_action
    def switch_applications(self, app_code: str):
        """Switch to a different application that is already open
        Args:
            app_code:str the code name of the application to switch to from the provided list of open applications
        """
        if self.grounding.platform == "darwin":
            return (
                "import pyautogui; import time; "
                "pyautogui.hotkey('command', 'space', interval=0.5); "
                f"pyautogui.typewrite({repr(app_code)}); "
                "pyautogui.press('enter'); time.sleep(1.0)"
            )
        if self.grounding.platform == "linux":
            return UBUNTU_APP_SETUP.replace("APP_NAME", app_code)
        if self.grounding.platform == "windows":
            return (
                "import pyautogui; import time; "
                "pyautogui.hotkey('win', 'd', interval=0.5); "
                f"pyautogui.typewrite({repr(app_code)}); "
                "pyautogui.press('enter'); time.sleep(1.0)"
            )
        raise ValueError(
            "Unsupported platform: "
            f"{self.grounding.platform}. Supported platforms are: darwin, linux, windows."
        )

    @tool_action
    def open(self, app_or_filename: str):
        """Open any application or file with name app_or_filename. Use this action to open applications or files on the desktop, do not open manually.
        Args:
            app_or_filename:str, the name of the application or filename to open
        """
        if self.grounding.platform == "linux":
            return (
                "import pyautogui; pyautogui.hotkey('win'); time.sleep(0.5); "
                f"pyautogui.write({repr(app_or_filename)}); time.sleep(1.0); "
                "pyautogui.hotkey('enter'); time.sleep(0.5)"
            )
        if self.grounding.platform == "darwin":
            return (
                "import pyautogui; import time; "
                "pyautogui.hotkey('command', 'space', interval=0.5); "
                f"pyautogui.typewrite({repr(app_or_filename)}); "
                "pyautogui.press('enter'); time.sleep(1.0)"
            )
        if self.grounding.platform == "windows":
            return (
                "import pyautogui; import time; "
                "pyautogui.hotkey('win'); time.sleep(0.5); "
                f"pyautogui.write({repr(app_or_filename)}); time.sleep(1.0); "
                "pyautogui.press('enter'); time.sleep(0.5)"
            )
        raise ValueError(
            "Unsupported platform: "
            f"{self.grounding.platform}. Supported platforms are: darwin, linux, windows."
        )

    @tool_action
    def type(
        self,
        element_description: Optional[str] = None,
        text: str = "",
        overwrite: bool = False,
        enter: bool = False,
    ):
        """Type text into a specific element
        Args:
            element_description:str, a detailed description of which element to enter text in.
            text:str, the text to type
            overwrite:bool, Assign it to True if the text should overwrite the existing text, otherwise assign it to False. Using this argument clears all text in an element.
            enter:bool, Assign it to True if the enter key should be pressed after typing the text, otherwise assign it to False.
        """
        lines = ["import pyautogui"]
        sudo_password = self.grounding._get_client_password()
        lines.extend(
            [
                "try:",
                "    import pyperclip",
                "except ImportError:",
                "    import shlex",
                "    import subprocess",
                f"    sudo_password = {repr(sudo_password)}",
                "    sudo_password = shlex.quote(sudo_password)",
                "    subprocess.run(",
                "        f'echo {sudo_password} | sudo -S apt-get install -y xclip xsel',",
                "        shell=True,",
                "        check=True,",
                "    )",
                "    subprocess.check_call([subprocess.sys.executable, '-m', 'pip', 'install', 'pyperclip'])",
                "    import pyperclip",
                "",
            ]
        )

        if element_description is not None:
            coords1 = self.grounding.generate_coords(
                element_description, self.grounding.obs
            )
            x, y = self.grounding.resize_coordinates(coords1)
            lines.append(f"pyautogui.click({x}, {y})")

        if overwrite:
            lines.append(
                "pyautogui.hotkey("
                + repr("command" if self.grounding.platform == "darwin" else "ctrl")
                + ", 'a')"
            )
            lines.append("pyautogui.press('backspace')")

        has_unicode = any(ord(char) > 127 for char in text)
        needs_clipboard = has_unicode or any(char in "<>" for char in text)

        if needs_clipboard:
            if self.grounding.platform == "linux":
                lines.extend(
                    [
                        "if not pyperclip.is_available():",
                        "    import shlex",
                        "    import subprocess",
                        f"    sudo_password = {repr(sudo_password)}",
                        "    sudo_password = shlex.quote(sudo_password)",
                        "    subprocess.run(",
                        "        f'echo {sudo_password} | sudo -S apt-get install -y xclip xsel',",
                        "        shell=True,",
                        "        check=True,",
                        "    )",
                        "",
                    ]
                )
            lines.extend(
                [
                    "try:",
                    f"    pyperclip.copy({repr(text)})",
                    "    pyautogui.hotkey("
                    + repr("command" if self.grounding.platform == "darwin" else "ctrl")
                    + ", 'v')",
                    "except Exception:",
                    f"    pyautogui.write({repr(text)})",
                ]
            )
        else:
            lines.append(f"pyautogui.write({repr(text)})")

        if enter:
            lines.append("pyautogui.press('enter')")
        return "\n".join(lines)

    @tool_action
    def drag_and_drop(
        self, starting_description: str, ending_description: str, hold_keys: List = []
    ):
        """Drag from the starting description to the ending description
        Args:
            starting_description:str, a very detailed description of where to start the drag action.
            ending_description:str, a very detailed description of where to end the drag action.
            hold_keys:List list of keys to hold while dragging
        """
        coords1 = self.grounding.generate_coords(
            starting_description, self.grounding.obs
        )
        coords2 = self.grounding.generate_coords(ending_description, self.grounding.obs)
        x1, y1 = self.grounding.resize_coordinates(coords1)
        x2, y2 = self.grounding.resize_coordinates(coords2)

        command = "import pyautogui; "
        command += f"pyautogui.moveTo({x1}, {y1}); "
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1., button='left'); pyautogui.mouseUp(); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "
        return command

    @tool_action
    def highlight_text_span(
        self, starting_phrase: str, ending_phrase: str, button: str = "left"
    ):
        """Highlight a text span between a provided starting phrase and ending phrase. Use this to highlight words, lines, and paragraphs.
        Args:
            starting_phrase:str, the phrase that denotes the start of the text span you want to highlight. If you only want to highlight one word, just pass in that single word.
            ending_phrase:str, the phrase that denotes the end of the text span you want to highlight. If you only want to highlight one word, just pass in that single word.
        """

        coords1 = self.grounding.generate_text_coords(
            starting_phrase, self.grounding.obs, alignment="start"
        )
        coords2 = self.grounding.generate_text_coords(
            ending_phrase, self.grounding.obs, alignment="end"
        )
        x1, y1 = coords1
        x2, y2 = coords2

        command = "import pyautogui; "
        command += f"pyautogui.moveTo({x1}, {y1}); "
        command += f"pyautogui.dragTo({x2}, {y2}, duration=1., button='{button}'); pyautogui.mouseUp(); "
        return command

    @tool_action
    def set_cell_values(
        self, cell_values: Dict[str, Any], app_name: str, sheet_name: str
    ):
        """Use this to set individual cell values in a spreadsheet. For example, setting A2 to "hello" would be done by passing {"A2": "hello"} as cell_values. The sheet must be opened before this command can be used.
        Args:
            cell_values: Dict[str, Any], A dictionary of cell values to set in the spreadsheet. The keys are the cell coordinates in the format "A1", "B2", etc.
                Supported value types include: float, int, string, bool, formulas.
            app_name: str, The name of the spreadsheet application. For example, "Some_sheet.xlsx".
            sheet_name: str, The name of the sheet in the spreadsheet. For example, "Sheet1".
        """
        return SET_CELL_VALUES_CMD.format(
            cell_values=cell_values,
            app_name=app_name,
            sheet_name=sheet_name,
            sudo_password=self.grounding._get_client_password(),
        )

    @tool_action
    def scroll(self, element_description: str, clicks: int, shift: bool = False):
        """Scroll the element in the specified direction
        Args:
            element_description:str, a very detailed description of which element to enter scroll in.
            clicks:int, the number of clicks to scroll can be positive (up) or negative (down).
            shift:bool, whether to use shift+scroll for horizontal scrolling
        """
        coords1 = self.grounding.generate_coords(
            element_description, self.grounding.obs
        )
        x, y = self.grounding.resize_coordinates(coords1)

        if shift:
            return (
                "import pyautogui; import time; "
                f"pyautogui.moveTo({x}, {y}); time.sleep(0.5); pyautogui.hscroll({clicks})"
            )
        return (
            "import pyautogui; import time; "
            f"pyautogui.moveTo({x}, {y}); time.sleep(0.5); pyautogui.vscroll({clicks})"
        )

    @tool_action
    def hotkey(self, keys: List):
        """Press a hotkey combination
        Args:
            keys:List the keys to press in combination in a list format (e.g. ['ctrl', 'c'])
        """
        keys = [f"'{key}'" for key in keys]
        return f"import pyautogui; pyautogui.hotkey({', '.join(keys)})"

    @tool_action
    def hold_and_press(self, hold_keys: List, press_keys: List):
        """Hold a list of keys and press a list of keys
        Args:
            hold_keys:List, list of keys to hold
            press_keys:List, list of keys to press in a sequence
        """
        press_keys_str = "[" + ", ".join([f"'{key}'" for key in press_keys]) + "]"
        command = "import pyautogui; "
        for k in hold_keys:
            command += f"pyautogui.keyDown({repr(k)}); "
        command += f"pyautogui.press({press_keys_str}); "
        for k in hold_keys:
            command += f"pyautogui.keyUp({repr(k)}); "

        return command

    @tool_action
    def wait(self, time: float):
        """Wait for a specified amount of time
        Args:
            time:float the amount of time to wait in seconds
        """
        return f"import time; time.sleep({time})"

    @tool_action
    def done(self):
        """End the current task with a success."""
        return "DONE"

    @tool_action
    def fail(self):
        """End the current task when it cannot be completed."""
        return "FAIL"

    @tool_action
    def report_infeasible(self, reason: str, evidence: str):
        """
        当你经过诊断确认任务无法完成时调用此工具。   

        Args:
            reason (str): 导致任务不可行的根本原因（例如："用户指定的目录为空" 或 "官方文档明确不支持该功能"）。
            evidence (str): 支持你判断的客观证据（例如："ls -la 命令返回 0 个文件" 或 "已查阅 VS Code 官方文档 URL..."）。
        """
        report = {
            "reason": (reason or "").strip(),
            "evidence": (evidence or "").strip(),
        }
        if hasattr(self.grounding, "set_infeasible_report"):
            self.grounding.set_infeasible_report(report)
        else:
            setattr(self.grounding, "last_infeasible_report", report)

        print(f"🚫 [Infeasible Report] Reason: {report['reason']}")
        print(f"🔍 [Evidence] {report['evidence']}")

        return "FAIL"
