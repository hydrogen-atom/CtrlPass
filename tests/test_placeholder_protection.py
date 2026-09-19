import re
from dataclasses import dataclass
from typing import List, Tuple

import pytest

# ---------------------------------------------------------------------------
# 与 document_processor.py 中完全一致的常量和模式
# ---------------------------------------------------------------------------
PLACEHOLDER_PATTERN = re.compile(r"<[CT]\d{4}>")
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
TABLE_PATTERN = re.compile(r"(?:^[ \t]*\|[^\n]*\|[ \t]*\n?){2,}", re.MULTILINE)


@dataclass
class ProtectedBlock:
    placeholder: str
    content: str
    block_type: str
    start_pos: int = 0
    end_pos: int = 0


# ---------------------------------------------------------------------------
# 核心逻辑副本（用户可直接运行，无需启动整个项目）
# ---------------------------------------------------------------------------

def _needs_protection(text: str) -> bool:
    if CODE_BLOCK_PATTERN.search(text):
        return True
    if TABLE_PATTERN.search(text):
        return True
    return False


def _extract_protected_blocks(text: str) -> Tuple[str, List[ProtectedBlock]]:
    patterns = [
        (CODE_BLOCK_PATTERN, "code", "C"),
        (TABLE_PATTERN, "table", "T"),
    ]

    matches = []
    for compiled_pattern, btype, prefix in patterns:
        for match in compiled_pattern.finditer(text):
            matches.append((match.start(), match.end(), match.group(), btype, prefix))

    matches.sort(key=lambda x: x[0])
    filtered = []
    last_end = -1
    for start, end, content, btype, prefix in matches:
        if start >= last_end:
            filtered.append((start, end, content, btype, prefix))
            last_end = end

    blocks: List[ProtectedBlock] = []
    counters = {"C": 0, "T": 0}
    new_text = text

    # 先生成所有占位符（按出现顺序），再从后往前替换，避免位置偏移
    placeholders = []
    for _start, _end, _content, _btype, prefix in filtered:
        idx = counters[prefix]
        placeholder = f"<{prefix}{idx:04d}>"
        counters[prefix] += 1
        placeholders.append(placeholder)

    for (start, end, content, btype, prefix), placeholder in zip(reversed(filtered), reversed(placeholders)):
        block = ProtectedBlock(
            placeholder=placeholder,
            content=content,
            block_type=btype,
            start_pos=start,
            end_pos=end,
        )
        blocks.insert(0, block)
        new_text = new_text[:start] + placeholder + new_text[end:]

    return new_text, blocks


def _restore_protected_blocks(chunks: List[str], blocks: List[ProtectedBlock]) -> List[str]:
    restored = []
    for text in chunks:
        for block in blocks:
            text = text.replace(block.placeholder, block.content)

        # 1. 清理完整占位符残留
        text = PLACEHOLDER_PATTERN.sub("", text)
        # 2. 清理被分块器切开的占位符碎片
        #    例如 <C000, 000>, <C000000> 等
        text = re.sub(r"<[CT]\d{0,4}", "", text)
        text = re.sub(r"\d{0,4}>", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        restored.append(text)
    return restored


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestNeedsProtection:
    def test_detects_code_block_with_backticks(self):
        text = "```python\nprint(1)\n```"
        assert _needs_protection(text) is True

    def test_detects_inline_code(self):
        text = "使用 `pip install` 命令"
        assert _needs_protection(text) is True

    def test_detects_markdown_table(self):
        text = "| 列A | 列B |\n| --- | --- |\n| 1   | 2   |"
        assert _needs_protection(text) is True

    def test_no_protection_needed_for_plain_text(self):
        text = "这是一段纯文本，没有任何代码块或表格。"
        assert _needs_protection(text) is False


class TestExtractProtectedBlocks:
    def test_single_code_block(self):
        code = "```python\nprint(1)\nprint(2)\n```"
        text = f"前文\n{code}\n后文"
        protected, blocks = _extract_protected_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].content == code
        assert blocks[0].block_type == "code"
        assert blocks[0].placeholder.startswith("<C")
        assert protected.count("```") == 0  # 原文中的 ``` 已被替换

    def test_multiple_code_blocks(self):
        code1 = "```bash\npip install numpy\n```"
        code2 = "`x + y`"
        text = f"A\n{code1}\nB\n{code2}\nC"
        protected, blocks = _extract_protected_blocks(text)

        assert len(blocks) == 2
        assert blocks[0].placeholder == "<C0000>"
        assert blocks[1].placeholder == "<C0001>"
        assert code1 not in protected
        assert code2 not in protected

    def test_table_protection(self):
        table = "| 姓名 | 年龄 |\n| ---- | ---- |\n| 张三 | 25   |"
        text = f"以下是表格：\n{table}\n结束"
        protected, blocks = _extract_protected_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].block_type == "table"
        assert blocks[0].placeholder.startswith("<T")
        assert "| 姓名 |" not in protected

    def test_mixed_code_and_table(self):
        code = "```python\nimport pandas\n```"
        table = "| A | B |\n| - | - |\n| 1 | 2 |"
        text = f"{code}\n\n{table}"
        protected, blocks = _extract_protected_blocks(text)

        assert len(blocks) == 2
        assert blocks[0].block_type == "code"
        assert blocks[1].block_type == "table"
        assert "<C0000>" in protected
        assert "<T0000>" in protected

    def test_no_nested_overlap(self):
        # 代码块内部包含类似表格的内容，不应被拆开
        code = "```python\n# | a | b |\nprint(1)\n```"
        text = code
        protected, blocks = _extract_protected_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].content == code


class TestRestoreProtectedBlocks:
    def test_restore_code_block(self):
        code = "```python\nprint('hello')\n```"
        text = f"前文\n{code}\n后文"
        protected, blocks = _extract_protected_blocks(text)

        # 模拟分块：在占位符之前切分，并模拟 20 字符 overlap
        split_pos = protected.find("<C0000>")
        chunk1 = protected[:split_pos]
        chunk2 = protected[max(0, split_pos - 20):]
        chunks = [chunk1, chunk2]

        restored = _restore_protected_blocks(chunks, blocks)
        full = "\n".join(restored)

        assert code in full
        assert "<C" not in full  # 无残留占位符

    def test_cleans_fragmented_placeholders(self):
        # 人为制造碎片场景（占位符被分块器拦腰切断）
        blocks = [ProtectedBlock(placeholder="<C0000>", content="```\ncode\n```", block_type="code")]
        chunks = ["<C000", "000>"]  # 碎片

        restored = _restore_protected_blocks(chunks, blocks)
        full = "".join(restored)

        assert "<C0000>" not in full  # 清理了碎片
        assert "<C000" not in full
        assert "000>" not in full
        assert full.strip() == ""  # 只剩空白

    def test_restores_multiple_blocks(self):
        code1 = "```bash\necho 1\n```"
        code2 = "```bash\necho 2\n```"
        text = f"A\n{code1}\nB\n{code2}\nC"
        protected, blocks = _extract_protected_blocks(text)

        # 模拟分块：直接按原样还原（未切割）
        restored = _restore_protected_blocks([protected], blocks)

        assert restored[0].count("```") == 4  # 两个代码块共 4 个 ```
        assert code1 in restored[0]
        assert code2 in restored[0]


class TestEndToEnd:
    """端到端测试：模拟真实分链路的 extract → fake_split → restore"""

    def _fake_split(self, text: str, chunk_size: int = 80) -> List[str]:
        """模拟 RecursiveCharacterTextSplitter 的硬切行为"""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            # 模拟 overlap
            if i > 0:
                chunk = text[max(0, i - 20) : i] + chunk
            chunks.append(chunk)
        return chunks

    def test_code_block_integrity_after_split(self):
        text = """# 安装指南

使用以下命令安装依赖：

```bash
pip install numpy pandas torch transformers
pip install matplotlib seaborn
```

然后运行程序。
"""
        protected, blocks = _extract_protected_blocks(text)
        chunks = self._fake_split(protected, chunk_size=60)
        restored = _restore_protected_blocks(chunks, blocks)

        full = "\n".join(restored)
        # 断言代码块完整存在
        assert "pip install numpy pandas torch transformers" in full
        assert "pip install matplotlib seaborn" in full
        # 断言代码块没有被切成 ```bash\npip install 这种碎片
        assert "```bash\npip install" not in full or "\n```" in full.split("```bash")[1][:100]

    def test_table_integrity_after_split(self):
        text = """参数说明：

| 参数名   | 类型   | 默认值 | 说明           |
| -------- | ------ | ------ | -------------- |
| lr       | float  | 0.001  | 学习率         |
| epochs   | int    | 100    | 训练轮数       |
| batch    | int    | 32     | 批大小         |

请根据上表配置模型。
"""
        protected, blocks = _extract_protected_blocks(text)
        chunks = self._fake_split(protected, chunk_size=50)
        restored = _restore_protected_blocks(chunks, blocks)

        full = "\n".join(restored)
        # 表格应完整出现
        assert "| 参数名   | 类型   | 默认值 | 说明           |" in full
        assert "| lr       | float  | 0.001  | 学习率         |" in full
        # 不应出现 | --- 碎片
        assert PLACEHOLDER_PATTERN.search(full) is None

    def test_plain_text_regression(self):
        """纯文本不应受任何影响"""
        text = "第一章\n\n这是正文。这是正文。这是正文。\n\n第二章\n\n更多正文。"
        assert not _needs_protection(text)
        # extract 仍可调用，但应返回空 blocks
        protected, blocks = _extract_protected_blocks(text)
        assert len(blocks) == 0
        assert protected == text


# ---------------------------------------------------------------------------
# 如果用户没有 pytest，也支持直接 python test_placeholder_protection.py 运行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
