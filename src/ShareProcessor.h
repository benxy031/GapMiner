/**
 * Header file of GapMiners pow processing functionality
 *
 * Copyright (C)  2014  The Gapcoin developers  <info@gapcoin.org>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __SHARE_PROCESSOR_H__
#define __SHARE_PROCESSOR_H__
#ifndef __STDC_FORMAT_MACROS 
#define __STDC_FORMAT_MACROS 
#endif
#ifndef __STDC_LIMIT_MACROS  
#define __STDC_LIMIT_MACROS  
#endif
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <queue>
#include "PoWCore/src/PoWProcessor.h"
#include "PoWCore/src/PoW.h"
#include "BlockHeader.h"

using namespace std;

class ShareProcessor : PoWProcessor {

  public:

    /* return the only instance of this */
    static ShareProcessor *get_processor();

    /* run clean up */
    static void cleanup();

    /**
     * Validates that a given PoW isn't obsolete, and if not adds it to the shares queue.
     * 
     * Validation includes Baillie-PSW probable primality test of gap endpoints.
     * See PRIMALITY-TESTING.md in project root for:
     * - Why BPSW is sufficient for mining
     * - Optional enhancements for record submissions
     * - Proof generation for competitive records
     * - Future optimization opportunities (Lucas tests, deterministic bases)
     */
    bool process(PoW *pow);

    /* update the current block header */
    void update_header(BlockHeader *header);

  private:

    /* private constructor to allow only one instance */
    ShareProcessor();

    /* private destructor */
    ~ShareProcessor();


    /* this will be a singleton */
    static ShareProcessor *only_instance;

    /* the current block header */
    BlockHeader *header;

    /* waiting share queue */
    queue<BlockHeader *> shares;

    /* indicates if this was initialized */
    static bool initialized;

    /* indicates that the share thread is running */
    static bool running;

    /* thread object for the share process thread */
    pthread_t thread;

    /* synchronization mutexes */
    static pthread_mutex_t queue_mutex;       
    static pthread_mutex_t wait_mutex;   
    static pthread_mutex_t creation_mutex;

    /* thread which processes the share queue */
    static void *share_processor(void *args);

    /**
     * test if share is for the current block
     * if true, returns the adjusted Block,
     * else NULL
     */
    BlockHeader *get_header_from(PoW *share);

};

#endif /* __SHARE_PROCESSOR_H__ */
