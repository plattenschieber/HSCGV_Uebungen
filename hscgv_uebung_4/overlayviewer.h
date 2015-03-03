/** \file
 * \brief QOSGViewer with 2D overlays
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Stefan Zellmann <zellmans@uni-koeln.de> and
 * Martin Aumueller <aumueller@uni-koeln.de>
 */

#ifndef OVERLAYVIEWER_H
#define OVERLAYVIEWER_H

#include "qosgviewer.h"

#include <QString>

//! Splitscreen viewer. Optionally show a little map. Display messages using QPainter.
class OverlayViewer : public QSplitscreenViewer
{
    Q_OBJECT
    public:
        //! Public constructor.
        OverlayViewer(QWidget* parent = 0, const QGLWidget* shareWidget = 0, Qt::WindowFlags f = 0);

        //! Display info messages.
        void showInfo(const QString& text, const double duration = 5.0);

        //! Draw text using QPainter.
        void drawText(QPainter *painter, QString text, double scale,
                double x, double y, double w, double h,
                bool background=true) const;

    protected:
        //! Draw time, score and info.
        virtual void paintOverlay(QPainter *painter);

    private:
        //! Draw an info text.
        void drawInfo(QPainter *painter) const;

        //! Info string.
        QString m_info;
        //! Info string animation start.
        double m_infoStart;
        //! Info string animation end.
        double m_infoEnd;
};

#endif
