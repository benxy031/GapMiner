# GapMiner - A standalone Gapcoin (GAP) CPU, GPU, rpc, pool miner
---
<br/>
## So what's the purpose of a standalone rpc miner? 


  * GapMiner is completely open source (GPLv3) and (hopefully)
    well documented, giving developers a good start into the
    pGap algorithm (Gapcoins prime gap based hashing algorithm).

  * simplicity and little dependencies (pure C++ code and only 
    pthread, openssl, curl, jansson, boost_system, gmp and mpfr dependencies).

  * speed (at the moment, GapMiner has no speed improvements in comparison with Gapcoin,
    but this is hopefully going to change.)



## get it running
---

First of all, keep in mind that GapMiner still has alpha qualities and 
doesn't claim to be the fastest CPU miner out there. The focal point
is the readability and understandability of the Gapcoin mining algorithm!

Plus, currently it's only for Linux. Sorry.

### required libraries
  - pthread
  - openssl
  - curl
  - jansson
  - gmp 
  - mpfr
  - boost system library

### installation
```sh
  git clone https://github.com/gapcoin-project/GapMiner.git
  cd GapMiner
  git submodule init
  git submodule update
  make all
  make install
```
## Usage
---

  `gapminer [--options]`

### basic

 - `-o  --host [ipv4]` host ip address

 - `-p  --port [port]` port to connect to

 - `-u  --user [user]` user for gapcoin rpc authentification

 - `-x  --pwd [pwd]` password for gapcoin rpc authentification

#### example:

`gapminer -o 127.0.0.1 -p 31397 -u rpcuser -x rpcpassword`

### advanced

 - `-q  --quiet` be quiet (only prints shares)

 - `-i  --stats-interval [NUM]` interval (sec) to print mining informations

 - `-t  --threads [NUM]` number of mining threads

 - `-l  --pull-interval [NUM]` seconds to wait between getwork request

 - `-s  --sieve-size [NUM]` the prime sieve size

 - `-r  --sieve-primes [NUM]` number of primes to sieve

 - `-f  --shift [NUM]` the adder shift

 - `-h  --help` print this information

 - `-v  --license` show license of this program
