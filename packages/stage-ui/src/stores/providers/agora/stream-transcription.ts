/**
 * Agora Real-Time Transcription (RTT) streaming implementation
 *
 * Architecture:
 * 1. Create Agora RTC client and join a temporary channel
 * 2. Publish microphone audio to the channel
 * 3. Call REST API to start STT agent for that channel
 * 4. STT agent joins, subscribes to audio, and publishes transcripts as data messages
 * 5. Client receives transcripts via stream-message events
 * 6. Transcripts are converted to StreamTranscriptionDelta streams
 */

import type { StreamTranscriptionDelta, StreamTranscriptionResult } from '@xsai/stream-transcription'

import type { AgoraSTTCredentials } from './rest-api'

import { agoraSTTJoin, agoraSTTLeave } from './rest-api'

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void
  let _isResolved = false
  let _isRejected = false
  const promise = new Promise<T>((res, rej) => {
    resolve = (value) => {
      _isResolved = true
      res(value)
    }
    reject = (reason) => {
      _isRejected = true
      rej(reason)
    }
  })

  return {
    promise,
    resolve,
    reject,
    get isResolved() { return _isResolved },
    get isRejected() { return _isRejected },
    set isResolved(value: boolean) { _isResolved = value },
    set isRejected(value: boolean) { _isRejected = value },
  }
}

/** STT data message from Agora STT agent (JSON protocol) */
interface AgoraSTTMessage {
  /** Unique identifier for this recognition segment */
  uid?: number
  /** Vendor identifier (typically 0) */
  vendor?: number
  /** Version of the message protocol */
  version?: number
  /** Sequence number for ordering */
  seqnum?: number
  /** Whether this is a final (not interim) result */
  is_final?: boolean
  /** The recognized text */
  text?: string
  /** Confidence score 0-1 */
  confidence?: number
  /** Sentence index */
  text_ts?: number
  /** Duration of the recognized speech in ms */
  duration_ms?: number
  /** Array of word-level results */
  words?: Array<{
    text: string
    start_ms: number
    duration_ms: number
    is_final: boolean
  }>
  /** Translated results */
  trans?: Array<{
    lang: string
    texts: string[]
    is_final: boolean
  }>
}

export interface AgoraStreamTranscriptionExtraOptions {
  credentials: AgoraSTTCredentials
  language?: string
  abortSignal?: AbortSignal
  /** UID for the local user in the RTC channel */
  localUid?: string
  /** UID for the STT subscriber bot */
  subBotUid?: string
  /** UID for the STT publisher bot */
  pubBotUid?: string
}

export interface AgoraStreamTranscriptionOptions {
  baseURL?: string | URL
  model?: string
  fetch?: typeof globalThis.fetch
  headers?: Record<string, string>
  abortSignal?: AbortSignal
  inputAudioStream?: ReadableStream<ArrayBuffer>
  file?: Blob
  credentials?: AgoraSTTCredentials
  language?: string
  localUid?: string
  subBotUid?: string
  pubBotUid?: string
}

export function streamAgoraTranscription(options: AgoraStreamTranscriptionOptions): StreamTranscriptionResult {
  const deferredText = createDeferred<string>()

  let text = ''
  let textStreamCtrl: ReadableStreamDefaultController<string> | undefined
  let fullStreamCtrl: ReadableStreamDefaultController<StreamTranscriptionDelta> | undefined

  const fullStream = new ReadableStream<StreamTranscriptionDelta>({
    start(controller) {
      fullStreamCtrl = controller
    },
  })

  const textStream = new ReadableStream<string>({
    start(controller) {
      textStreamCtrl = controller
    },
  })

  const doStream = async () => {
    const credentials = options.credentials
    if (!credentials?.appId || !credentials.customerId || !credentials.customerSecret) {
      throw new Error('Agora STT credentials (appId, customerId, customerSecret) are required.')
    }

    const language = options.language || 'en-US'
    const localUid = options.localUid || String(Math.floor(Math.random() * 100000) + 1000)
    const subBotUid = options.subBotUid || '2'
    const pubBotUid = options.pubBotUid || '3'
    const channelName = `airi-stt-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

    // Dynamically import Agora RTC SDK (browser-only)
    const AgoraRTC = (await import('agora-rtc-sdk-ng')).default

    const client = AgoraRTC.createClient({ mode: 'rtc', codec: 'vp8' })
    let agentId: string | undefined
    let micTrack: ReturnType<typeof AgoraRTC.createMicrophoneAudioTrack> extends Promise<infer T> ? T : never

    const cleanup = async () => {
      try {
        if (agentId) {
          await agoraSTTLeave(credentials, agentId).catch(err =>
            console.warn('Agora STT leave error:', err),
          )
          agentId = undefined
        }
      }
      catch {}
      try {
        if (micTrack) {
          micTrack.stop()
          micTrack.close()
        }
      }
      catch {}
      try {
        await client.leave()
      }
      catch {}
    }

    // Handle abort signal
    if (options.abortSignal) {
      if (options.abortSignal.aborted) {
        await cleanup()
        const error = new DOMException('Aborted', 'AbortError')
        throw error
      }
      options.abortSignal.addEventListener('abort', async () => {
        console.info('Agora STT: abort signal received, cleaning up...')
        await cleanup()

        if (!deferredText.isResolved && !deferredText.isRejected) {
          const doneDelta: StreamTranscriptionDelta = { type: 'transcript.text.done', delta: '' }
          try {
            fullStreamCtrl?.enqueue(doneDelta)
            fullStreamCtrl?.close()
            textStreamCtrl?.close()
          }
          catch {}
          deferredText.resolve(text)
        }
      })
    }

    // Listen for data messages from STT publisher bot
    client.on('stream-message', (_uid: number, data: Uint8Array) => {
      try {
        const jsonStr = new TextDecoder().decode(data)
        const msg: AgoraSTTMessage = JSON.parse(jsonStr)

        if (msg.text && msg.is_final) {
          const transcript = msg.text.trim()
          if (transcript) {
            text += `${transcript} `
            const delta: StreamTranscriptionDelta = {
              type: 'transcript.text.delta',
              delta: transcript,
            }
            fullStreamCtrl?.enqueue(delta)
            textStreamCtrl?.enqueue(transcript)
            console.info('Agora STT transcribed (final):', transcript)
          }
        }
        else if (msg.text) {
          console.info('Agora STT transcribed (interim):', msg.text)
        }
      }
      catch (err) {
        console.warn('Agora STT: failed to parse data message:', err)
      }
    })

    client.on('exception', (event) => {
      console.warn('Agora RTC exception:', event)
    })

    // Join RTC channel
    console.info('Agora STT: joining channel', channelName, 'as UID', localUid)
    await client.join(credentials.appId, channelName, null, Number(localUid))
    console.info('Agora STT: joined channel successfully')

    // Create and publish microphone track
    micTrack = await AgoraRTC.createMicrophoneAudioTrack({
      encoderConfig: 'speech_standard',
    })
    await client.publish([micTrack])
    console.info('Agora STT: microphone track published')

    // Start STT agent via REST API
    const joinResponse = await agoraSTTJoin(credentials, {
      languages: [language],
      maxIdleTime: 60,
      rtcConfig: {
        channelName,
        subBotUid,
        pubBotUid,
        subscribeAudioUids: [localUid],
        enableJsonProtocol: true,
      },
    })

    agentId = joinResponse.agent_id
    console.info('Agora STT: agent started', agentId, 'status:', joinResponse.status)
  }

  doStream().catch((err) => {
    console.error('Agora STT stream error:', err)
    const error = err instanceof Error ? err : new Error(String(err))
    try {
      fullStreamCtrl?.error(error)
      textStreamCtrl?.error(error)
    }
    catch {}
    if (!deferredText.isResolved && !deferredText.isRejected) {
      deferredText.reject(error)
    }
  })

  return {
    fullStream,
    textStream,
    text: deferredText.promise,
  }
}
