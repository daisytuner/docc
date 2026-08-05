import { Hono } from 'hono'
import * as fs from 'fs'
import path from 'path';


const router = new Hono()

router.post('/transfertune_sdfg', async (c) => {
    // Optional testing mode: if paths are provided via headers, load JSONs from disk.
    const filePath = path.resolve(__dirname, '../../../res/matmul');
    const hint = c.req.header('RPC-Hint') ?? filePath

    try {
        const sdfgResPath = c.req.header('SDFG-Result-Path') ?? hint +'.sdfg.json'
        const replayPath = c.req.header('SDFG-Replay-Path') ?? hint +'.replay.json'

        let sdfg = null;
        let replay = null;
        if (fs.existsSync(sdfgResPath)) {
            sdfg = JSON.parse(fs.readFileSync(sdfgResPath, 'utf-8'))
            console.info(`Read SDFG from ${sdfgResPath}`)
        } else {
            console.warn(`SDFG result file not found at ${sdfgResPath}`)
        }
        if (fs.existsSync(replayPath)) {
            replay = JSON.parse(fs.readFileSync(replayPath, 'utf-8'))
            console.info(`Read LocalReplay from ${replayPath}`)
        } else {
            console.warn(`LocalReplay result file not found at ${replayPath}`)
        }

        return c.json({
            sdfg_result: sdfg? { sdfg: sdfg } : undefined,
            local_replay: replay ?? undefined,
            metadata: {
                region_id: 'test-region',
                speedup: 1.0,
                vector_distance: 0.0,
            }
        })
    } catch (err) {
        console.error('Failed to load testing JSON files:', err)
        return c.json({ error: 'Failed to load testing JSON files' }, 500)
    }

    // Default behaviour: fall back to hard-coded matmul example.
    return c.json({ error: 'No match found' })
})

export default router
