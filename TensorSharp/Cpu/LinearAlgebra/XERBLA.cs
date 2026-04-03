using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorSharp.Cpu.LinearAlgebra
{
    public class XERBLA
    {

        public XERBLA()
        {

        }

        /// <summary>
        /// Purpose
        /// =======
        /// 
        /// XERBLA  is an error handler for the LAPACK routines.
        /// It is called by an LAPACK routine if an input parameter has an
        /// invalid value.  A message is printed and execution stops.
        /// 
        /// Installers may consider modifying the STOP statement in order to
        /// call system-specific exception-handling facilities.
        /// 
        ///</summary>
        /// <param name="SRNAME">
        /// (input) CHARACTER*6
        /// The name of the routine which called XERBLA.
        ///</param>
        /// <param name="INFO">
        /// (input) INTEGER
        /// The position of the invalid parameter in the parameter list
        /// of the calling routine.
        ///</param>
        public void Run(string SRNAME, int INFO)
        {

            #region Strings

            SRNAME = SRNAME.Substring(0, 6);

            #endregion


            #region Prolog

            // *
            // *  -- LAPACK auxiliary routine (version 3.1) --
            // *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
            // *     November 2006
            // *
            // *     .. Scalar Arguments ..
            // *     ..
            // *
            // *  Purpose
            // *  =======
            // *
            // *  XERBLA  is an error handler for the LAPACK routines.
            // *  It is called by an LAPACK routine if an input parameter has an
            // *  invalid value.  A message is printed and execution stops.
            // *
            // *  Installers may consider modifying the STOP statement in order to
            // *  call system-specific exception-handling facilities.
            // *
            // *  Arguments
            // *  =========
            // *
            // *  SRNAME  (input) CHARACTER*6
            // *          The name of the routine which called XERBLA.
            // *
            // *  INFO    (input) INTEGER
            // *          The position of the invalid parameter in the parameter list
            // *          of the calling routine.
            // *
            // * =====================================================================
            // *
            // *     .. Executable Statements ..
            // *

            #endregion

            //ERROR-ERROR      WRITE( *, FMT = 9999 )SRNAME, INFO;
            // *
            return;
            // *
            // *
            // *     End of XERBLA
            // *
        }
    }
}
