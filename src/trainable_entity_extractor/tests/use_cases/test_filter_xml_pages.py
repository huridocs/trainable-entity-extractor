from unittest import TestCase

from trainable_entity_extractor.use_cases.FilterValidSegmentsPages import FilterValidSegmentsPagesUseCase


class TestFilterXmlPages(TestCase):
    def test_filter_xml_pages(self):
        xml_text = """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE pdf2xml SYSTEM "pdf2xml.dtd">
        
        <pdf2xml producer="poppler" version="23.07.0">
        <page number="1" position="absolute" top="0" left="0" height="792" width="612">
            <fontspec id="0" size="9" family="TimesNewRomanPSMT" color="#000000"/>
            <fontspec id="1" size="14" family="TimesNewRomanPSMT" color="#000000"/>
        <text top="45" left="123" width="87" height="16" font="1">United Nations </text>
        <text top="39" left="495" width="15" height="22" font="2">A</text>
        </page>
        <page number="2" position="absolute" top="0" left="0" height="792" width="612">
            <fontspec id="14" size="9" family="TimesNewRomanPS" color="#000000"/>
        <text top="53" left="60" width="43" height="8" font="14"><b>A/INF/76/1 </b></text>
        <text top="57" left="306" width="2" height="8" font="0"> </text>
        <text top="67" left="60" width="2" height="8" font="0"> </text>
        </page>
        
        <page number="3" position="absolute" top="0" left="0" height="792" width="612">
            <fontspec id="15" size="10" family="TimesNewRomanPS" color="#000000"/>
        <text top="53" left="60" width="43" height="8" font="14"><b>Page 3</b></text>
        </page>
        </pdf2xml>
        """

        xml_expect = """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE pdf2xml SYSTEM "pdf2xml.dtd">
        
        <pdf2xml producer="poppler" version="23.07.0">
        <page number="1" position="absolute" top="0" left="0" height="792" width="612">
            <fontspec id="0" size="9" family="TimesNewRomanPSMT" color="#000000"/>
            <fontspec id="1" size="14" family="TimesNewRomanPSMT" color="#000000"/>
        </page>
        <page number="2" position="absolute" top="0" left="0" height="792" width="612">
            <fontspec id="14" size="9" family="TimesNewRomanPS" color="#000000"/>
        <text top="53" left="60" width="43" height="8" font="14"><b>A/INF/76/1 </b></text>
        <text top="57" left="306" width="2" height="8" font="0"> </text>
        <text top="67" left="60" width="2" height="8" font="0"> </text>
        </page>
        
        <page number="3" position="absolute" top="0" left="0" height="792" width="612">
            <fontspec id="15" size="10" family="TimesNewRomanPS" color="#000000"/>
        </page>
        </pdf2xml>
        """

        result = FilterValidSegmentsPagesUseCase.filter_xml_pages(xml_text, [2]).split()
        self.assertEqual(xml_expect.split(), result)

    def test_filter_xml_pages_one_line(self):
        xml_text = """<foo />
        <pdf2xml producer="poppler" version="23.07.0"><page number="1" position="absolute" top="0" left="0" height="792" width="612">
            <fontspec id="0" size="9" family="TimesNewRomanPSMT" color="#000000"/>
            <fontspec id="1" size="14" family="TimesNewRomanPSMT" color="#000000"/>
        <text top="45" left="123" width="87" height="16" font="1">United Nations </text>
        <text top="39" left="495" width="15" height="22" font="2">A</text>
        </page><page number="2" position="absolute" top="0" left="0" height="792" width="612">
            <fontspec id="14" size="9" family="TimesNewRomanPS" color="#000000"/>
        <text top="53" left="60" width="43" height="8" font="14"><b>A/INF/76/1 </b></text>
        <text top="57" left="306" width="2" height="8" font="0"> </text>
        <text top="67" left="60" width="2" height="8" font="0"> </text>
        </page><page number="3" position="absolute" top="0" left="0" height="792" width="612">
            <fontspec id="15" size="10" family="TimesNewRomanPS" color="#000000"/>
        <text top="53" left="60" width="43" height="8" font="14"><b>Page 3</b></text>
        </page>
        </pdf2xml>
        """

        xml_expect = """<foo />
        <pdf2xml producer="poppler" version="23.07.0"><page number="1" position="absolute" top="0" left="0" height="792" width="612">
            <fontspec id="0" size="9" family="TimesNewRomanPSMT" color="#000000"/>
            <fontspec id="1" size="14" family="TimesNewRomanPSMT" color="#000000"/>
        </page><page number="2" position="absolute" top="0" left="0" height="792" width="612">
            <fontspec id="14" size="9" family="TimesNewRomanPS" color="#000000"/>
        <text top="53" left="60" width="43" height="8" font="14"><b>A/INF/76/1 </b></text>
        <text top="57" left="306" width="2" height="8" font="0"> </text>
        <text top="67" left="60" width="2" height="8" font="0"> </text>
        </page><page number="3" position="absolute" top="0" left="0" height="792" width="612">
            <fontspec id="15" size="10" family="TimesNewRomanPS" color="#000000"/>
        </page>
        </pdf2xml>
        """

        self.assertEqual(xml_expect.split(), FilterValidSegmentsPagesUseCase.filter_xml_pages(xml_text, [2]).split())

    def test_filter_xml_pages_two_pages(self):
        xml_text = """<foo />
                <pdf2xml producer="poppler" version="23.07.0"><page number="1" position="absolute" top="0" left="0" height="792" width="612">
                    <fontspec id="0" size="9" family="TimesNewRomanPSMT" color="#000000"/>
                <text top="45" left="123" width="87" height="16" font="1">United Nations </text>
                <text top="39" left="495" width="15" height="22" font="2">A</text>
                </page><page number="2" position="absolute" top="0" left="0" height="792" width="612">
                    <fontspec id="14" size="9" family="TimesNewRomanPS" color="#000000"/>
                <text top="53" left="60" width="43" height="8" font="14"><b>A/INF/76/1 </b></text>
                <text top="57" left="306" width="2" height="8" font="0"> </text>
                <text top="67" left="60" width="2" height="8" font="0"> </text>
                </page><page number="3" position="absolute" top="0" left="0" height="792" width="612">
                    <fontspec id="15" size="10" family="TimesNewRomanPS" color="#000000"/>
                <text top="53" left="60" width="43" height="8" font="14"><b>Page 3</b></text>
                </page>
                </pdf2xml>
                """

        xml_expect = """<foo />
                <pdf2xml producer="poppler" version="23.07.0"><page number="1" position="absolute" top="0" left="0" height="792" width="612">
                    <fontspec id="0" size="9" family="TimesNewRomanPSMT" color="#000000"/>
                <text top="45" left="123" width="87" height="16" font="1">United Nations </text>
                <text top="39" left="495" width="15" height="22" font="2">A</text>
                </page><page number="2" position="absolute" top="0" left="0" height="792" width="612">
                    <fontspec id="14" size="9" family="TimesNewRomanPS" color="#000000"/>
                <text top="53" left="60" width="43" height="8" font="14"><b>A/INF/76/1 </b></text>
                <text top="57" left="306" width="2" height="8" font="0"> </text>
                <text top="67" left="60" width="2" height="8" font="0"> </text>
                </page><page number="3" position="absolute" top="0" left="0" height="792" width="612">
                    <fontspec id="15" size="10" family="TimesNewRomanPS" color="#000000"/>
                </page>
                </pdf2xml>
                """

        self.assertEqual(xml_expect.split(), FilterValidSegmentsPagesUseCase.filter_xml_pages(xml_text, [1, 2]).split())

    def test_filter_xml_pages_last_page(self):
        xml_text = """<foo />
                        <pdf2xml producer="poppler" version="23.07.0"><page number="1" position="absolute" top="0" left="0" height="792" width="612">
                            <fontspec id="0" size="9" family="TimesNewRomanPSMT" color="#000000"/>
                        <text top="45" left="123" width="87" height="16" font="1">United Nations </text>
                        <text top="39" left="495" width="15" height="22" font="2">A</text>
                        </page><page number="2" position="absolute" top="0" left="0" height="792" width="612">
                            <fontspec id="14" size="9" family="TimesNewRomanPS" color="#000000"/>
                        <text top="53" left="60" width="43" height="8" font="14"><b>A/INF/76/1 </b></text>
                        <text top="57" left="306" width="2" height="8" font="0"> </text>
                        <text top="67" left="60" width="2" height="8" font="0"> </text>
                        </page><page number="3" position="absolute" top="0" left="0" height="792" width="612">
                            <fontspec id="15" size="10" family="TimesNewRomanPS" color="#000000"/>
                        <text top="53" left="60" width="43" height="8" font="14"><b>Page 3</b></text>
                        </page>
                        </pdf2xml>
                        """

        xml_expect = """<foo />
                        <pdf2xml producer="poppler" version="23.07.0"><page number="1" position="absolute" top="0" left="0" height="792" width="612">
                            <fontspec id="0" size="9" family="TimesNewRomanPSMT" color="#000000"/>
                        </page><page number="2" position="absolute" top="0" left="0" height="792" width="612">
                            <fontspec id="14" size="9" family="TimesNewRomanPS" color="#000000"/>
                        </page><page number="3" position="absolute" top="0" left="0" height="792" width="612">
                            <fontspec id="15" size="10" family="TimesNewRomanPS" color="#000000"/>
                        <text top="53" left="60" width="43" height="8" font="14"><b>Page 3</b></text>
                        </page>
                        </pdf2xml>
                        """

        self.assertEqual(xml_expect.split(), FilterValidSegmentsPagesUseCase.filter_xml_pages(xml_text, [3]).split())

    def test_filter_xml_pages_id_with_zeros(self):
        xml_text = """<foo />
                      <page number="001" position="absolute" top="0" left="0" height="792" width="612">
                                    <fontspec id="0" size="9" family="TimesNewRomanPSMT" color="#000000"/>
                                <text top="45" left="123" width="87" height="16" font="1">United Nations </text>
                                <text top="39" left="495" width="15" height="22" font="2">A</text>
                                </page><page number="02" position="absolute" top="0" left="0" height="792" width="612">
                                    <fontspec id="14" size="9" family="TimesNewRomanPS" color="#000000"/>
                                <text top="53" left="60" width="43" height="8" font="14"><b>A/INF/76/1 </b></text>
                                <text top="57" left="306" width="2" height="8" font="0"> </text>
                                <text top="67" left="60" width="2" height="8" font="0"> </text>
                                </page>
                                </pdf2xml>
                                """

        xml_expect = """<foo />
                      <page number="001" position="absolute" top="0" left="0" height="792" width="612">
                                    <fontspec id="0" size="9" family="TimesNewRomanPSMT" color="#000000"/>
                                </page><page number="02" position="absolute" top="0" left="0" height="792" width="612">
                                    <fontspec id="14" size="9" family="TimesNewRomanPS" color="#000000"/>
                                <text top="53" left="60" width="43" height="8" font="14"><b>A/INF/76/1 </b></text>
                                <text top="57" left="306" width="2" height="8" font="0"> </text>
                                <text top="67" left="60" width="2" height="8" font="0"> </text>
                                </page>
                                </pdf2xml>
                                """

        self.assertEqual(xml_expect.split(), FilterValidSegmentsPagesUseCase.filter_xml_pages(xml_text, [2]).split())

    def test_filter_xml_should_do_nothing_when_not_matching_tags(self):
        xml_text_more_opening_ones = """<foo />
       <page number="2">
       <text top="53" left="60" width="43" height="8" font="14"></page></text>
       </page>
       <page number="3"> </page>
        <foo />"""

        xml_text_more_opening_ones_2 = """<foo />
       <page number="2">
       <text top="53" left="60" width="43" height="8" font="14"></text>
       </page>
       <page number="3"> </page>
       <page number="4">
        <foo />"""

        xml_text_more_closing = """<foo />
       <page number="2">
       <text top="53" left="60" width="43" height="8" font="14"></page></text>
       </page>
       <page number="3"> </page>
       <page number="4">
        <foo />"""

        self.assertEqual(
            xml_text_more_opening_ones.split(),
            FilterValidSegmentsPagesUseCase.filter_xml_pages(xml_text_more_opening_ones, [2]).split(),
        )

        self.assertEqual(
            xml_text_more_opening_ones_2.split(),
            FilterValidSegmentsPagesUseCase.filter_xml_pages(xml_text_more_opening_ones_2, [2]).split(),
        )

        self.assertEqual(
            xml_text_more_closing.split(),
            FilterValidSegmentsPagesUseCase.filter_xml_pages(xml_text_more_closing, [2]).split(),
        )
