from html.parser import HTMLParser

from forte.data.readers.html_reader import ForteHTMLParser


class CustomHTMLParser(ForteHTMLParser):
    def __init__(self):
        super().__init__()

        self._start_tags = []
        self._end_tags = []

        self._recent_start = ()

    def handle_starttag(self, tag, attrs):
        self._start_tags.append(
            (tag, self.getpos(), self.get_starttag_text())
        )
        self._recent_start = self.getpos()

    def handle_endtag(self, tag):
        # TODO: skip self closing tag.
        if not self._recent_start == self.getpos():
            self._end_tags.append((tag, self.getpos()))

    def get_tags(self):
        return self._start_tags, self._end_tags

    def reset(self):
        super().reset()
        self._start_tags = []
        self._end_tags = []
