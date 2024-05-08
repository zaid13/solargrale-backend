import pandas as pd
import matplotlib
from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from fpdf import FPDF


def addLogogo(pdfLink,file_name,lenthOfObservationPoints,):
    pdf = FPDF()
    # pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('arial', 'B', 12)

    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('arial', 'B', 12)
    pdf.cell(60)
    pdf.image('assets/logo.png', x = 10, y = None, w = 100, h = 0, type = 'png', link = '')

    # pdf.cell(75, 10, "SOLAR CALC GLARE ANALYSIS", 0, 2, 'C')

    ctr=1
    for i in range(0,lenthOfObservationPoints):

        pdf.image('assets/'+file_name+f'barchart{i+1}.png', x = 50, y = None, w = 100, h = 120, type = 'png', link = '')
        # pdf.cell(4)

    pdf.output(pdfLink, 'F')