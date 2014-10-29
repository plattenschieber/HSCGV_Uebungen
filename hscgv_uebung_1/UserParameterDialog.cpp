/** \file
 * \brief Implementation of the UserParameter dialog 
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 1 - "Ui, OpenGL!"
 *
 * Created by Christian Vogelgsang <Vogelgsang@informatik.uni-erlangen.de>,
 * changed by Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "UserParameterDialog.h"

// create a new UserParameter dialog
UserParameterDialog::UserParameterDialog(QWidget *parent)
: QDialog(parent)
{
    ui.setupUi(this);
    saveState();

    connect(ui.userSpin,SIGNAL(valueChanged(double)),SIGNAL(userParameter(double)));

    restoreState();
}


// destroy the UserParameter dialog
UserParameterDialog::~UserParameterDialog()
{
    // deletion of QObjects is done automatically
}

// convert the internal state of the dialog to the GUI representation
void UserParameterDialog::restoreState()
{
    ui.userSpin->setValue(m_saveUserParameter);
    emit userParameter(ui.userSpin->value());
}


// convert the GUI state to the internal one
void UserParameterDialog::saveState()
{
    m_saveUserParameter = ui.userSpin->value();
}

// restore state when dialog is cancelled
void UserParameterDialog::reject()
{
    restoreState();
    QDialog::reject();
}

// store state when dialog is shown
void UserParameterDialog::showEvent(QShowEvent * event) 
{
    Q_UNUSED(event);
    saveState();
}
