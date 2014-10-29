/** \file
 * \brief Model loading and drawing
 *
 * Programmierpraktikum Computergrafik (CGP): Aufgabe 1 - "Ui, OpenGL!"
 *
 * Created by Martin Aumueller <aumueller@uni-koeln.de>
 */

#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <QObject>

#include <GL/glew.h>

//! internal state of PLY loader
struct PlyFile;

// ----- Model -----

//! load and draw a PLY model
// this should render the model in a way corresponding to the current options
class Model : public QObject
{
    // include meta object information
    Q_OBJECT
public:

        //! load a model from file
        Model(QObject *parent=NULL);
        //! dtor
        virtual ~Model();
        //! read a model from file
        void read(const QString &filename);
        //! get coordinates of center of bounding box of model
        void getCenter(float *x, float *y, float *z) const;
        //! return size of bounding box of model
        GLfloat size() const;

        //! draw using traditional glBegin()/glEnd()
        void drawImmediate(bool smooth);
        //! draw using vertex arrays
        void drawArray(bool smooth);
        //! draw using vertex buffer objects (VBOs)
        void drawVBO(bool smooth);

signals:
        //! show some information to the user
        void info(QString msg);

protected:
        // ----- internal state -----

        // current model data

        //! one vertex of the model
        struct Vertex
        {
            //! x coordinate
            float x;
            //! y coordinate
            float y;
            //! z coordinate
            float z;
            //! x-component of vertex normal
            float nx;
            //! y-component of vertex normal
            float ny;
            //! z-component of vertex normal
            float nz;
            //! unused
            void *other_props;
        };


        //! one face of the model
        struct Face
        {
            //! number of vertices in face
            unsigned char nverts;
            //! pointer to vertex index list
            int *verts;
            //! x-component of face normal
            float nx;
            //! y-component of face normal
            float ny;
            //! z-component of face normal
            float nz;
            //! unused
            void *other_props;
        };

        //! list of vertices
        std::vector<Vertex> m_vert;
        //! list of faces
        std::vector<Face> m_face;

        //! reference to PLY file
        PlyFile *m_ply;
        //! center of model bounding box
        GLfloat m_center[3];
        //! length of diagonal of model bounding box
        GLfloat m_size;

        //! compute smooth vertex normals
        void computeVertexNormals();
        //! compute data structures for vertex array rendering
        void computeVertexArrayData();
        //! buffer vertex array data for VBO rendering
        void bufferArrayData();
        //! release all allocated resources
        void release();

        // vertex array data
        //! vertices
        std::vector<GLfloat> m_vertexArray;
        //! vertex normals
        std::vector<GLfloat> m_vertexNormalArray;
        //! face normals
        std::vector<GLfloat> m_faceNormalArray;
        //! vertex indices
        std::vector<GLint> m_vertexIndexArray;
        //! sizes of primitives (polygons)
        std::vector<GLint> m_primitiveSizeArray;
        //! array of pointers into m_vertexIndexArray to first index of each primitive (vertex array rendering)
        std::vector<GLvoid *> m_vertexIndexStartArray;
        //! array of offsets into m_vertexIndexArray to first index of each primitive (VBO rendering)
        std::vector<GLvoid *> m_primitiveOffsetArray;

        //! indices into VBO ids
        enum { BO_VERTEX=0, BO_VERTEX_NORMAL, BO_FACE_NORMAL, BO_VERTEX_INDEX, NUM_BOS };
        //! VBO ids
        GLuint m_bo[NUM_BOS];
};

#endif
