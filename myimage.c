#include <Python.h>
#include <math.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

#define DEG2RAD (NPY_PI / 180.0)

static double deg2rad(double degree)
{
    return degree * DEG2RAD;
}

double even(double x)
{
    const double r = round(x);
    const double d = r - x;

    if ((d != 0.5f) && (d != -0.5f))
    {
        return r;
    }

    if (fmod(r, 2.0) == 0.0)
    {
        return r;
    }

    return x - d;
}

static PyObject *hough_transform(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *image, *theta_range, *reference_image;
    PyArrayObject *image_array, *theta_range_array, *reference_image_array;
    theta_range = Py_None;
    reference_image = Py_None;
    static char *kwlist[] = {"image", "reference_image", "theta_range", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|OO", kwlist, &image, &reference_image, &theta_range))
    {
        fprintf(stderr, "Invalid argument\n");
        Py_RETURN_NONE;
    }
    image_array = (PyArrayObject *) PyArray_FROM_O(image);
    int use_reference = 0;
    if (reference_image != Py_None)
    {
        use_reference = 1;
        reference_image_array = (PyArrayObject *) PyArray_FROM_O(reference_image);
    }
    if (theta_range != Py_None)
    {
        theta_range_array = (PyArrayObject *) PyArray_FROM_OT(theta_range, NPY_FLOAT64);
    }
    else
    {
        npy_intp theta_dims[] = {2};
        theta_range_array = PyArray_SimpleNew(1, theta_dims, NPY_FLOAT64);
        *(npy_float64 *) PyArray_GETPTR1(theta_range_array, 0) = -30;
        *(npy_float64 *) PyArray_GETPTR1(theta_range_array, 1) = 31;
    }

    PyObject *thetas = PyArray_Arange(*(npy_float64 *) PyArray_GETPTR1(theta_range_array, 0),
                                      *(npy_float64 *) PyArray_GETPTR1(theta_range_array, 1), 1.0, NPY_FLOAT64);
    int thetas_size = PyArray_Size(thetas);
    npy_intp trig_dims[] = {thetas_size};
    PyObject *sin_table = PyArray_ZEROS(1, trig_dims, NPY_FLOAT64, 0);
    PyObject *cos_table = PyArray_ZEROS(1, trig_dims, NPY_FLOAT64, 0);
    npy_float64 *thetas_data = (npy_float64 *) PyArray_DATA((PyArrayObject *) thetas);
    npy_float64 *sin_table_data = (npy_float64 *) PyArray_DATA((PyArrayObject *) sin_table);
    npy_float64 *cos_table_data = (npy_float64 *) PyArray_DATA((PyArrayObject *) cos_table);
    for (int i = 0; i < thetas_size; i++)
    {
        thetas_data[i] = deg2rad(thetas_data[i]);
        sin_table_data[i] = sin(thetas_data[i]);
        cos_table_data[i] = cos(thetas_data[i]);
    }
    int height = PyArray_DIM(image_array, 0);
    int width = PyArray_DIM(image_array, 1);
    double distance = ceil(sqrt(width * width + height * height));
    PyObject *rhos = PyArray_Arange(-distance, distance, 1.0, NPY_FLOAT32);
    npy_intp accumulator_dims[] = {PyArray_Size(rhos), PyArray_Size(thetas)};
    PyObject *accumulator = PyArray_ZEROS(2, accumulator_dims, NPY_UINT64, 0);
    PyObject *zero = PyLong_FromLong(0);
    PyObject *equal = PyObject_RichCompare((PyObject *) image_array, zero, Py_EQ);
    Py_DECREF(zero);
    PyObject *equal_array = PyArray_FROM_O(equal);
    PyObject *nonzero = PyArray_Nonzero((PyArrayObject *) equal_array);
    Py_DECREF(equal_array);
    Py_DECREF(equal);
    PyObject *y_indexes = PyTuple_GetItem(nonzero, 0);
    PyObject *x_indexes = PyTuple_GetItem(nonzero, 1);
    npy_uint64 *accumulator_data = (npy_uint64 *) PyArray_DATA((PyArrayObject *) accumulator);

    int x_indexes_size = PyArray_Size(x_indexes);
    for (int i = 0; i < x_indexes_size; i++)
    {
        npy_int64 x = *(npy_int64 *) PyArray_GETPTR1((PyArrayObject *) x_indexes, i);
        npy_int64 y = *(npy_int64 *) PyArray_GETPTR1((PyArrayObject *) y_indexes, i);
        for (int theta_index = 0; theta_index < thetas_size; theta_index++)
        {
            int rho = even(x * cos_table_data[theta_index] + y * sin_table_data[theta_index] + distance);
            accumulator_data[(rho * thetas_size) + theta_index] += !use_reference ? 1 : 255 -
                                                                                        (*(npy_uint8 *) PyArray_GETPTR2(
                                                                                                reference_image_array,
                                                                                                y, x));
        }
    }

    Py_DECREF(sin_table);
    Py_DECREF(cos_table);
    Py_DECREF(nonzero);
    Py_DECREF(image_array);
    Py_DECREF(theta_range_array);
    if (use_reference) Py_DECREF(reference_image_array);

    return Py_BuildValue("NNN", accumulator, thetas, rhos);
}

static PyObject *calc_threshold_by_otsu(PyObject *self, PyObject *args, PyObject *keywds)
{
    PyObject *image, *histogram, *class1, *class2, *n1, *n2, *w1, *w2, *inner1, *inner2;
    PyArrayObject *histogram_array;
    static char *kwlist[] = {"image", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &image))
    {
        fprintf(stderr, "Invalid argument\n");
        Py_RETURN_NONE;
    }
    histogram = PyObject_CallMethod(image, "histogram", "");
    histogram_array = (PyArrayObject *) PyArray_FROM_O(histogram);

    double s_max = -1.0;
    int threshold = 0;
    for (int t = 0; t < 256; t++)
    {
        class1 = PySequence_GetSlice((PyObject *) histogram_array, 0, t);
        class2 = PySequence_GetSlice((PyObject *) histogram_array, t, PySequence_Length((PyObject *) histogram_array));

        n1 = PyArray_Sum((PyArrayObject *) class1, NPY_MAXDIMS, PyArray_DESCR((PyArrayObject *) class1)->type_num,
                         NULL);
        n2 = PyArray_Sum((PyArrayObject *) class2, NPY_MAXDIMS, PyArray_DESCR((PyArrayObject *) class2)->type_num,
                         NULL);
        w1 = PyArray_Arange(0, t, 1.0, NPY_FLOAT32);
        w2 = PyArray_Arange(t, 256, 1.0, NPY_FLOAT32);
        inner1 = PyArray_InnerProduct(w1, class1);
        inner2 = PyArray_InnerProduct(w2, class2);
        long dn1 = PyLong_AsLong(n1);
        long dn2 = PyLong_AsLong(n2);
        double mean1 = dn1 != 0 ? PyFloat_AsDouble(inner1) / dn1 : 0;
        double mean2 = dn2 != 0 ? PyFloat_AsDouble(inner2) / dn2 : 0;
        double s = dn1 * dn2 * pow(mean1 - mean2, 2.0);
        if (s > s_max)
        {
            s_max = s;
            threshold = t;
        }
        Py_DECREF(inner2);
        Py_DECREF(inner1);
        Py_DECREF(w2);
        Py_DECREF(w1);
        Py_DECREF(class2);
        Py_DECREF(class1);
        Py_DECREF(n2);
        Py_DECREF(n1);
    }

    Py_DECREF(histogram);
    Py_DECREF(histogram_array);

    return Py_BuildValue("i", threshold);
}

static PyObject *binarization(PyObject *self, PyObject *args, PyObject *keywds)
{
    static char *kwlist[] = {"image", "threshold", NULL};
    PyArrayObject *image_array;
    int threshold;
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oi", kwlist, &image_array, &threshold))
    {
        fprintf(stderr, "Invalid argument\n");
        Py_RETURN_NONE;
    }

    int height = PyArray_DIM(image_array, 0);
    int width = PyArray_DIM(image_array, 1);

    npy_intp dims[] = {height, width};
    PyObject *dest_image = PyArray_ZEROS(2, dims, NPY_UBYTE, 0);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            npy_ubyte value = *(npy_ubyte *) PyArray_GETPTR2(image_array, y, x);
            if (value > threshold)
            {
                *(npy_ubyte *) PyArray_GETPTR2((PyArrayObject *) dest_image, y, x) = 255;
            }
        }
    }

    return Py_BuildValue("N", dest_image);;
}

static PyMethodDef myimage_module_methods[] = {
        {
                "hough_transform",
                      (PyCFunction) hough_transform,
                            METH_VARARGS | METH_KEYWORDS,
                               "Hough transform\n"
                               "  Args:\n"
                               "      image(ndarray): Input image.\n"
                               "      reference_image(ndarray): Image referenced to luminance value.\n"
                               "      theta_range(tuple): Range of theta to be calculated.\n"
                               "  Returns:\n"
                               "      ('accumulator, thetas, rhos) (tuple)\n"
                               "      accumulator(ndarray): Number of votes in ρ-θ space.\n"
                               "      thetas(ndarray): Array of angles (radians).\n"
                               "      rhos(ndarray): Array of distance.\n"
        },
        {
                "calc_threshold_by_otsu",
                      (PyCFunction) calc_threshold_by_otsu,
                            METH_VARARGS | METH_KEYWORDS,
                               "Calc threshold by otsu\n"
                               "  Args:\n"
                               "      image(ndarray): Input image.\n"
                               "  Returns:\n"
                               "      threshold(int): Thresholds calculated by Otsu's threshold determination method.\n"
        },
        {
                "binarization",
                      (PyCFunction) binarization,
                            METH_VARARGS | METH_KEYWORDS,
                               "Binarization\n"
                               "  Args:\n"
                               "      image(ndarray): Input image.\n"
                               "      threshold(int): Threshold used for binarization.\n"
                               "  Returns:\n"
                               "      dest_image(ndarray): Binarized image.\n"
        },
        {       NULL, NULL, 0, NULL}
};

static struct PyModuleDef myimage_module_definition = {
        PyModuleDef_HEAD_INIT,
        "myimage",
        "Extension module that provides some function for image",
        -1,
        myimage_module_methods
};

PyMODINIT_FUNC PyInit_myimage(void)
{
    Py_Initialize();

    import_array();

    return PyModule_Create(&myimage_module_definition);
}

