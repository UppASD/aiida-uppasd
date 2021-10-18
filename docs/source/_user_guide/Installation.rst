Installation
============================

You can download UppASD-AiiDA from our github repository first with command:

.. code-block::

    git clone https://github.com/UppASD/aiida-uppasd.git

Then you could install the plugin into your |:computer:| directly or within some Python environment with pip:

.. code-block::
    
    pip install -e .

After the installation is finished, you need to use reentry to update AiiDA's entry point (before that please make sure you have install and set the aiida profile correctly):

.. code-block::

    reentry scan

    
We highly recommand you to restart your AiiDA daemon at the same time with:

.. code-block::

    verdi daemon restart

|:stuck_out_tongue_winking_eye:| Congratulations! Now you could enjoy your time with AiiDA-UppASD interface! 


