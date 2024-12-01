from setuptools import setup , find_packages
import LunarAI
#https://blog.csdn.net/crown_0726/article/details/119192289
setup(
    name='LunarAI' ,
    version=LunarAI.__version__,
    description=('An Neural Network tool'),
    long_description='An Neural Network tool',
    long_description_content_type = 'text/markdown',
    author='Haozhe Xu',
    author_email='2779630178@qq.com',
    maintainer='Haozhe Xu',
    maintainer_email='2779630178@qq.com',
    license='MIT License',
    packages=find_packages(),
    platforms=['all'],
    url='https://github.com/Haozhe-py/LunarAI',
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        #'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries'
    ]
)