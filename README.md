# Rational Model of Categorization

Also known as Dirichlet process mixture model

This rational model of categorization is proposed by Anderson (1991) and
extended by Sanborn, Griffiths, and Navarro (2010). This model is identical to
Dirichlet process mixture model with either local MAP or particle filter
approximations. More details are found in the papers listed below.

- Anderson, J. R. (1991). The adaptive nature of human categorization.
  Psychological Review, 98, 409-429.

- Sanborn, A. N., Griffiths, T. L., & Navarro, D. J. (2010).  Rational
  approximations to rational models: alternative algorithms for category
  learning. Psychological Review, 117, 1144-1167.


## Files

- dpmm: main script for the model.
- main: Example usage of dpmm.
- test_dpmm: simulates the model and compares the results against what has been reported.
- libdpmm_r: interface for R. Example usage is in script.R


## Dependencies

- C++11 standard libraries


## Note

- Testing fails at SGN Figure 11 replication. The local MAP performs worse, and
  the particle filter model performs better, than what has been reported.
  Given all the other tests the model passes, it is not clear to me why this happens.
  Currently there is no plan to fix this.


## Licence

Copyright (c) 2015 Takao Noguchi (tkngch@runbox.com)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
