/** \file
 * \brief Help dialog
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Stefan Zellmann <zellmans@uni-koeln.de> and
 * Martin Aumueller <aumueller@uni-koeln.de>
 */

#ifndef HELPDIALOG_H
#define HELPDIALOG_H

#include <QDialog>

//! Help dialog, explain keyboard controls.
class HelpDialog : public QDialog
{
    Q_OBJECT
    public:
        //! Build a QDialog to display keyboard controls.
        HelpDialog(QWidget* parent = 0);
};

#endif
