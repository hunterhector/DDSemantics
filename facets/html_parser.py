from html.parser import HTMLParser, starttagopen, charref, entityref, incomplete
from html import unescape
from typing import re


class CustomHTMLParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)

        self.start_tags = []
        self.end_tags = []

        self.last_append = []

    def handle_starttag(self, tag, attrs):
        self.start_tags.append((tag, (self.getpos()[0], self.getpos()[1]),))

        self.last_append = self.start_tags

    def handle_endtag(self, tag):
        self.end_tags.append((tag, (self.getpos()[0], self.getpos()[1]),))

        self.last_append = self.end_tags

    def get_endpos(self):
        self.last_append[-1] = self.last_append[-1] + (
        (self.getpos()[0], self.getpos()[1]),)

    def get_tags(self):
        return self.start_tags, self.end_tags

    def reset(self):
        super().reset()
        self.start_tags = []
        self.end_tags = []



# Internal -- handle data as far as reasonable.  May leave state
# and data to be processed by a subsequent call.  If 'end' is
# true, force handling all data as if followed by EOF marker.
def goahead(self, end):
    rawdata = self.rawdata
    i = 0
    n = len(rawdata)
    while i < n:
        if self.convert_charrefs and not self.cdata_elem:
            j = rawdata.find('<', i)
            if j < 0:
                # if we can't find the next <, either we are at the end
                # or there's more text incoming.  If the latter is True,
                # we can't pass the text to handle_data in case we have
                # a charref cut in half at end.  Try to determine if
                # this is the case before proceeding by looking for an
                # & near the end and see if it's followed by a space or ;.
                amppos = rawdata.rfind('&', max(i, n - 34))
                if (amppos >= 0 and
                        not re.compile(r'[\s;]').search(rawdata, amppos)):
                    break  # wait till we get all the text
                j = n
        else:
            match = self.interesting.search(rawdata, i)  # < or &
            if match:
                j = match.start()
            else:
                if self.cdata_elem:
                    break
                j = n
        if i < j:
            if self.convert_charrefs and not self.cdata_elem:
                self.handle_data(unescape(rawdata[i:j]))
            else:
                self.handle_data(rawdata[i:j])
        i = self.updatepos(i, j)
        if i == n: break
        startswith = rawdata.startswith
        if startswith('<', i):
            if starttagopen.match(rawdata, i):  # < + letter
                k = self.parse_starttag(i)
            elif startswith("</", i):
                k = self.parse_endtag(i)
            elif startswith("<!--", i):
                k = self.parse_comment(i)
            elif startswith("<?", i):
                k = self.parse_pi(i)
            elif startswith("<!", i):
                k = self.parse_html_declaration(i)
            elif (i + 1) < n:
                self.handle_data("<")
                k = i + 1
            else:
                break
            if k < 0:
                if not end:
                    break
                k = rawdata.find('>', i + 1)
                if k < 0:
                    k = rawdata.find('<', i + 1)
                    if k < 0:
                        k = i + 1
                else:
                    k += 1
                if self.convert_charrefs and not self.cdata_elem:
                    self.handle_data(unescape(rawdata[i:k]))
                else:
                    self.handle_data(rawdata[i:k])
            i = self.updatepos(i, k)
            self.get_endpos()  # only modification: gets end position of tags
        elif startswith("&#", i):
            match = charref.match(rawdata, i)
            if match:
                name = match.group()[2:-1]
                self.handle_charref(name)
                k = match.end()
                if not startswith(';', k - 1):
                    k = k - 1
                i = self.updatepos(i, k)
                continue
            else:
                if ";" in rawdata[i:]:  # bail by consuming &#
                    self.handle_data(rawdata[i:i + 2])
                    i = self.updatepos(i, i + 2)
                break
        elif startswith('&', i):
            match = entityref.match(rawdata, i)
            if match:
                name = match.group(1)
                self.handle_entityref(name)
                k = match.end()
                if not startswith(';', k - 1):
                    k = k - 1
                i = self.updatepos(i, k)
                continue
            match = incomplete.match(rawdata, i)
            if match:
                # match.group() will contain at least 2 chars
                if end and match.group() == rawdata[i:]:
                    k = match.end()
                    if k <= i:
                        k = n
                    i = self.updatepos(i, i + 1)
                # incomplete
                break
            elif (i + 1) < n:
                # not the end of the buffer, and can't be confused
                # with some other construct
                self.handle_data("&")
                i = self.updatepos(i, i + 1)
            else:
                break
        else:
            assert 0, "interesting.search() lied"
    # end while
    if end and i < n and not self.cdata_elem:
        if self.convert_charrefs and not self.cdata_elem:
            self.handle_data(unescape(rawdata[i:n]))
        else:
            self.handle_data(rawdata[i:n])
        i = self.updatepos(i, n)
    self.rawdata = rawdata[i:]


CustomHTMLParser.goahead = goahead


