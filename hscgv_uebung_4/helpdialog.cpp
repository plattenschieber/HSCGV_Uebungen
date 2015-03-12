/** \file
 * \brief Help dialog
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created by Stefan Zellmann <zellmans@uni-koeln.de> and
 * Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "helpdialog.h"

#include <QBoxLayout>
#include <QLabel>
#include <QTextEdit>
#include <QTextStream>

HelpDialog::HelpDialog(QWidget* parent)
: QDialog(parent)
{
    setWindowTitle("Keyboard Controls");
    QLayout *layout = new QVBoxLayout(this);

    QTextEdit* textbox = new QTextEdit(this);
    textbox->setReadOnly(true);
    textbox->setFocusPolicy(Qt::NoFocus);
    QString helptext;
    QTextStream ts(&helptext);
    ts << "Cursor Left: select previous slice axis\n";
    ts << "Cursor Right: select next slice axis\n";
    ts << "Cursor Up: move selected slice forward\n";
    ts << "Cursor Down: move selected slice back\n";
    ts << "\n";
    ts << "c: switch to CPU\n";
    ts << "g: switch to GPU\n";
    ts << "\n";
    ts << "l: toggle line display\n";
    ts << "s: toggle slice display\n";
    ts << "\n";
    ts << "d: colorize according to density\n";
    ts << "v: colorize according to velocity\n";
    ts << "\n";
    ts << "0-9: start simulation with different conditions\n";
    ts << "p: pause simulation\n";
    ts << ".: advance simulation by one step\n";
    textbox->setText(helptext);
    layout->addWidget(textbox);
    resize(400, 400);
}
