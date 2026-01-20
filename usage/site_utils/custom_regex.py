import re
import os
from typing import List, Tuple


class RegexFilter:
    """Enhanced regex filter with support for letter substitution"""

    def __init__(
        self,
        whitelist_file="matchlist.txt",
        whitelist_no_regex_file="matchlist_no_regex.txt",
        blacklist_file="blacklist.txt",
    ):
        self.sub_letters = {
            "a": ["a", "@", "4"],
            "b": ["b", "8"],
            "c": ["c", "©"],
            "d": ["d", "Ð", "Ԑ"],
            "e": ["e", "3"],
            "f": ["f", "7"],
            "g": ["g", "9", "6"],
            "h": ["h", "Н"],
            "i": ["i", "l", "1", "!", "¡"],
            "k": ["k", "К"],
            "l": ["l", "1", "i", "!"],
            "m": ["m", "м"],
            "n": ["n", "И"],
            "o": ["o", "0", "Ø", "〇"],
            "r": ["r", "Я", "®"],
            "s": ["s", "5", "$", "𐌔"],
            "t": ["t", "7", "т"],
            "u": ["u", "Ц", "υ"],
            "v": ["v", "Ѵ", "ν"],
            "w": ["w", "Ш", "VV"],
            "x": ["x", "Х"],
            "y": ["y", "`/", "¥", "ү"],
            "z": ["z", "5"],
            "0": ["0", "O", "Ø", "〇"],
            "1": ["1", "I", "l", "!", "|"],
            "5": ["5", "S", "𐌔"],
            "8": ["8", "B", "⑧"],
            " ": [".*"],
        }

        self.whitelist = [self._gen_regex(p) for p in self._load_list(whitelist_file)]
        self.whitelist_no_regex = self._load_list(whitelist_no_regex_file)
        self.blacklist = self._load_list(blacklist_file)

    def _regex_or(self, lst: List[str]) -> str:
        return (
            f"({'|'.join(re.escape(item) if item != '.*' else item for item in lst)})"
        )

    def _gen_regex(self, s: str, i: int = 0) -> str:
        if i >= len(s):
            return s
        elif s[i].lower() in self.sub_letters:
            front = s[:i]
            back = s[i + 1 :]
            regex_sub = self._regex_or(self.sub_letters[s[i].lower()])
            newstr = front + regex_sub + back
            return self._gen_regex(newstr, i + len(regex_sub))
        else:
            return self._gen_regex(s, i + 1)

    def _load_list(self, filename: str) -> List[str]:
        """Loads a list of strings from a file, creating if doesn't exist"""
        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8") as f:
                pass

        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def matches(self, text: str) -> Tuple[bool, str]:
        """
        Returns (accept, match_string) based on filter rules
        Priority: whitelist > blacklist > default accept
        """
        # Check whitelist (both no_regex and regex versions)
        for pattern in self.whitelist_no_regex:
            if re.search(re.escape(pattern), text, re.IGNORECASE):
                return True, pattern

        for pattern in self.whitelist:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return True, match.group()

        # Check blacklist
        for pattern in self.blacklist:
            if re.search(re.escape(pattern), text, re.IGNORECASE):
                return False, ""

        return True, ""
