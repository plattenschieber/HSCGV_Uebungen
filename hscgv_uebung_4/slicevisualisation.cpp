/** \file
 * \brief Slice visualisation
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 4 - "LBM"
 *
 * Created Martin Aumueller <aumueller@uni-koeln.de>
 */

#include "slicevisualisation.h"
#include "slice.h"
#include "lbm.h"

#include <osg/LightModel>
#include <osg/PositionAttitudeTransform>
#include <osg/Geode>

#include <cassert>

SliceVisualisation::SliceVisualisation(LBMD3Q19 *lbm)
: m_lbm(lbm)
{
    // the three slices
    for(int i=0; i<3; ++i)
    {
        m_sliceNum.push_back(m_lbm->getDimension(i) / 2);
        m_slicePosition.push_back(new osg::PositionAttitudeTransform);
        addChild(m_slicePosition.back());

        osg::Geode *geode = new osg::Geode;
        m_slicePosition.back()->addChild(geode);
        m_slice.push_back(new Slice(m_lbm, i));
        geode->addDrawable(m_slice.back());

        osg::StateSet *state = geode->getOrCreateStateSet();
        osg::LightModel *lm = new osg::LightModel;
        lm->setTwoSided(true);
        state->setAttributeAndModes(lm, osg::StateAttribute::ON);
        geode->setStateSet(state);
    }
}

SliceVisualisation::~SliceVisualisation()
{
}

void SliceVisualisation::setSlice(int axis, int slice)
{
    assert(axis >= 0 && size_t(axis) < m_sliceNum.size());
    if(slice >= 0 && slice < m_lbm->getDimension(axis))
        m_sliceNum[axis] = slice;
}

int SliceVisualisation::getSlice(int axis) const
{
    assert(axis >= 0 && size_t(axis) < m_sliceNum.size());
    return m_sliceNum[axis];
}

void SliceVisualisation::update(Slice::DataValue value)
{
    for(size_t i=0; i<m_slice.size(); ++i)
    {
        switch(i%3)
        {
            case 0:
                m_slicePosition[i]->setPosition(osg::Vec3(m_sliceNum[i]+0.5, 0., 0.));
                m_slice[i]->update(m_sliceNum[i], value);
                break;
            case 1:
                m_slicePosition[i]->setPosition(osg::Vec3(0., m_sliceNum[i]+0.5, 0.));
                m_slice[i]->update(m_sliceNum[i], value);
                break;
            case 2:
                m_slicePosition[i]->setPosition(osg::Vec3(0., 0., m_sliceNum[i]+0.5));
                m_slice[i]->update(m_sliceNum[i], value);
                break;
        }
    }
}
