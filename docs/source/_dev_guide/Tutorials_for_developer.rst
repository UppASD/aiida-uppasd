=======================
Add your new output parser
=======================

You could find all parsers in /UppASD_AiiDA/core_parser.py file, once you find your output file is not parsered correctly you need to do THREE steps to add the new parser to your output file, the first one is open the /UppASD_AiiDA/core_calcs.py file and add one block followed "# output sections" in define, like here:



.. code-block::
class UppASD(CalcJob):
    .......
    ....
    ..
    def define(cls, spec):  
    ....
    ...
    .
        # output sections:
        spec.output('**', valid_type=**, required=False,
                    help='**')

Replaced all '**' placeholders and make your a new output port for new parser.

Then you could defined one parser function in core_parser file like:

.. code-block::
    
    def **_file_paser(self,file_name_of_**):
        """
        your help comments here
        """        
        result = pd.read_csv('**',
                             sep='\s+', header=None).drop([0])
        '**' = np.array(list(np.array(result)[:, 0]))
        return Iter_num_average, **

Still replace the placeholder and write your own code.
After that you need to use your parser function to parse the outputfile to AiiDA databse, add code block like:

.. code-block::
if '**' in name:
               **_filename = name
                # parse **.xx.out
                self.logger.info("Parsing '{}'".format(**_filename))
                with output_folder.open(**_filename, 'rb') as f:
                    **= self.**_file_paser(
                        f)
                    ** = ArrayData()
                    **.set_array('**', **)
                self.out('**', **)

Then you need go back to plugin's root dir re-install the plugin into your |:computer:| like use:

.. code-block::
    
    pip install -e .

And use reentry to update AiiDA's entry point:

.. code-block::

    reentry scan

    
We highly recommand you to restart your AiiDA daemon at the same time with:

.. code-block::

    verdi daemon restart