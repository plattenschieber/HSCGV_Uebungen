/** \file
 * \brief QOSGViewer with 2D overlays
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Stefan Zellmann <zellmans@uni-koeln.de> and
 * Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "overlayviewer.h"
#include "clock.h"

static const int Alpha = 128;
static const int Border = 4;

OverlayViewer::OverlayViewer(QWidget *parent, const QGLWidget *shareWidget, Qt::WindowFlags f)
: QSplitscreenViewer(parent, shareWidget, f)
, m_infoStart(-1.0)
, m_infoEnd(-1.0)
{
}

void OverlayViewer::paintOverlay(QPainter *painter)
{
    drawInfo(painter);
}

void OverlayViewer::drawText(QPainter *painter, QString text, const double scale,
        double x, double y, const double w, const double h,
        const bool background) const
{
    bool alignTop = y >= 0.;
    x = fabs(x);
    y = fabs(y);

    QFont fnt = font();
    fnt.setPointSizeF(scale*font().pointSizeF());
    QFontMetrics metrics = QFontMetrics(fnt);
    int border = qMax(Border, metrics.leading());
    QRect rect = metrics.boundingRect(width()*x , height()*y, width() - 2*border, int(height()*h),
            Qt::AlignCenter | Qt::TextWordWrap, text);

    painter->setRenderHint(QPainter::TextAntialiasing);
    painter->setPen(Qt::white);
    painter->setFont(fnt);
    double yy = height()*y;
    if(!alignTop)
        yy += height()*h - rect.height() - 2*border;
    if(background)
        painter->fillRect(QRect(width()*x, yy, width()*w, rect.height() + 2*border),
                QColor(0, 0, 0, Alpha));
    painter->drawText(width()*x+(width()*w - rect.width())/2, border + yy,
            rect.width(), rect.height(),
            Qt::AlignCenter | Qt::TextWordWrap, text);
    painter->setFont(font());
}

void OverlayViewer::drawInfo(QPainter *painter) const
{
    const double AnimTime = 0.5; // sliding animation
    const double Y = 0.8; // final y position
    const double now = Clock::now();
    if(now > m_infoEnd)
        return;

    double y = Y;
    if(now - m_infoStart < AnimTime)
        y = Y + (1.-Y)*(1.-(now-m_infoStart)/AnimTime);
    else if(m_infoEnd - now < AnimTime)
        y = Y + (1.-Y)*(1.-(m_infoEnd-now)/AnimTime);

    y *= -1.; // align to bottom
    drawText(painter, m_info, 1.0, 0.0, 0.0, 1.0, 0.1);
}

void OverlayViewer::showInfo(const QString& text, const double duration)
{
    m_info = text;
    const double now = Clock::now();
    if(m_infoEnd <= now)
        m_infoStart = now;
    m_infoEnd = now + duration;
}
