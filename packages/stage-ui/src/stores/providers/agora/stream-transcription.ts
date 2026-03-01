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

/** Word-level result from Agora STT protobuf (agora.audio2text.Word) */
interface AgoraSTTWord {
  text: string
  startMs?: number
  durationMs?: number
  isFinal?: boolean
  confidence?: number
}

/** Translation result from Agora STT protobuf (agora.audio2text.Translation) */
interface AgoraSTTTranslation {
  isFinal?: boolean
  lang?: string
  texts?: string[]
}

/**
 * STT data message from Agora STT agent (agora.audio2text.Text protobuf).
 * Also supports JSON protocol as fallback.
 */
interface AgoraSTTMessage {
  vendor?: number
  version?: number
  seqnum?: number
  uid?: number
  flag?: number
  time?: number
  lang?: number
  starttime?: number
  offtime?: number
  words?: AgoraSTTWord[]
  end_of_segment?: boolean
  duration_ms?: number
  data_type?: string
  trans?: AgoraSTTTranslation[]
  culture?: string
  textTs?: number
  /** Legacy: populated by JSON protocol or as convenience accessor */
  text?: string
  is_final?: boolean
}

/**
 * Parse Agora STT data message (supports both JSON and Protobuf formats).
 *
 * Protobuf schema: agora.audio2text.Text (fields 1-16)
 * Field 10 is `repeated Word` (submessage), NOT a plain string.
 */
function parseAgoraSTTData(data: Uint8Array): AgoraSTTMessage | null {
  // Try JSON first
  try {
    const str = new TextDecoder().decode(data)
    if (str.startsWith('{')) {
      return JSON.parse(str) as AgoraSTTMessage
    }
  }
  catch { /* not JSON, try protobuf */ }

  // Decode protobuf (agora.audio2text.Text)
  try {
    const msg = decodeProtobufText(data)
    // Build convenience `text` field from words
    if (msg.words?.length) {
      msg.text = msg.words.map(w => w.text).join('')
      msg.is_final = msg.end_of_segment ?? msg.words.every(w => w.isFinal)
    }
    console.info('Agora STT: decoded protobuf:', msg)
    return msg
  }
  catch (err) {
    console.warn('Agora STT: protobuf decode failed:', err)
    const hex = Array.from(data.slice(0, 64), b => b.toString(16).padStart(2, '0')).join(' ')
    console.warn('Agora STT: hex dump:', hex)
    return null
  }
}

// ── Minimal protobuf wire-format helpers ──

function pbReadVarint(buf: Uint8Array, state: { pos: number }): number {
  let result = 0
  let shift = 0
  while (state.pos < buf.length) {
    const byte = buf[state.pos++]
    result |= (byte & 0x7F) << shift
    if ((byte & 0x80) === 0)
      return result >>> 0 // unsigned
    shift += 7
  }
  return result >>> 0
}

function pbReadBytes(buf: Uint8Array, state: { pos: number }): Uint8Array {
  const len = pbReadVarint(buf, state)
  const slice = buf.slice(state.pos, state.pos + len)
  state.pos += len
  return slice
}

function pbReadString(buf: Uint8Array, state: { pos: number }): string {
  return new TextDecoder().decode(pbReadBytes(buf, state))
}

function pbSkipField(buf: Uint8Array, state: { pos: number }, wireType: number) {
  if (wireType === 0) {
    pbReadVarint(buf, state) // skip varint
  }
  else if (wireType === 2) {
    pbReadBytes(buf, state) // skip length-delimited
  }
  else if (wireType === 1) {
    state.pos += 8 // 64-bit fixed
  }
  else if (wireType === 5) {
    state.pos += 4 // 32-bit fixed
  }
  else {
    // Unknown wire type – stop parsing to avoid infinite loop
    state.pos = buf.length
  }
}

/** Decode agora.audio2text.Word submessage */
function decodeProtobufWord(buf: Uint8Array): AgoraSTTWord {
  const word: AgoraSTTWord = { text: '' }
  const state = { pos: 0 }
  while (state.pos < buf.length) {
    const tag = pbReadVarint(buf, state)
    const fieldNumber = tag >>> 3
    const wireType = tag & 0x07
    switch (fieldNumber) {
      case 1: // text (string)
        if (wireType === 2)
          word.text = pbReadString(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 2: // startMs (int32)
        if (wireType === 0)
          word.startMs = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 3: // durationMs (int32)
        if (wireType === 0)
          word.durationMs = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 4: // isFinal (bool)
        if (wireType === 0)
          word.isFinal = pbReadVarint(buf, state) === 1
        else pbSkipField(buf, state, wireType)
        break
      case 5: // confidence (double, wire type 1 = 64-bit)
        if (wireType === 1) {
          const view = new DataView(buf.buffer, buf.byteOffset + state.pos, 8)
          word.confidence = view.getFloat64(0, true)
          state.pos += 8
        }
        else { pbSkipField(buf, state, wireType) }
        break
      default:
        pbSkipField(buf, state, wireType)
    }
  }
  return word
}

/** Decode agora.audio2text.Translation submessage */
function decodeProtobufTranslation(buf: Uint8Array): AgoraSTTTranslation {
  const trans: AgoraSTTTranslation = { texts: [] }
  const state = { pos: 0 }
  while (state.pos < buf.length) {
    const tag = pbReadVarint(buf, state)
    const fieldNumber = tag >>> 3
    const wireType = tag & 0x07
    switch (fieldNumber) {
      case 1:
        if (wireType === 0)
          trans.isFinal = pbReadVarint(buf, state) === 1
        else pbSkipField(buf, state, wireType)
        break
      case 2:
        if (wireType === 2)
          trans.lang = pbReadString(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 3:
        if (wireType === 2)
          trans.texts!.push(pbReadString(buf, state))
        else pbSkipField(buf, state, wireType)
        break
      default:
        pbSkipField(buf, state, wireType)
    }
  }
  return trans
}

/** Decode agora.audio2text.Text top-level message */
function decodeProtobufText(buf: Uint8Array): AgoraSTTMessage {
  const msg: AgoraSTTMessage = {}
  const state = { pos: 0 }

  while (state.pos < buf.length) {
    const tag = pbReadVarint(buf, state)
    const fieldNumber = tag >>> 3
    const wireType = tag & 0x07

    switch (fieldNumber) {
      case 1: // vendor (int32)
        if (wireType === 0)
          msg.vendor = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 2: // version (int32)
        if (wireType === 0)
          msg.version = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 3: // seqnum (int32)
        if (wireType === 0)
          msg.seqnum = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 4: // uid (uint32)
        if (wireType === 0)
          msg.uid = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 5: // flag (int32)
        if (wireType === 0)
          msg.flag = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 6: // time (int64)
        if (wireType === 0)
          msg.time = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 7: // lang (int32)
        if (wireType === 0)
          msg.lang = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 8: // starttime (int32)
        if (wireType === 0)
          msg.starttime = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 9: // offtime (int32)
        if (wireType === 0)
          msg.offtime = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 10: // repeated Word words (submessage)
        if (wireType === 2) {
          const wordBuf = pbReadBytes(buf, state)
          if (!msg.words)
            msg.words = []
          msg.words.push(decodeProtobufWord(wordBuf))
        }
        else { pbSkipField(buf, state, wireType) }
        break
      case 11: // end_of_segment (bool)
        if (wireType === 0)
          msg.end_of_segment = pbReadVarint(buf, state) === 1
        else pbSkipField(buf, state, wireType)
        break
      case 12: // duration_ms (int32)
        if (wireType === 0)
          msg.duration_ms = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 13: // data_type (string)
        if (wireType === 2)
          msg.data_type = pbReadString(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 14: // repeated Translation trans (submessage)
        if (wireType === 2) {
          const transBuf = pbReadBytes(buf, state)
          if (!msg.trans)
            msg.trans = []
          msg.trans.push(decodeProtobufTranslation(transBuf))
        }
        else { pbSkipField(buf, state, wireType) }
        break
      case 15: // culture (string)
        if (wireType === 2)
          msg.culture = pbReadString(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      case 16: // textTs (int64)
        if (wireType === 0)
          msg.textTs = pbReadVarint(buf, state)
        else pbSkipField(buf, state, wireType)
        break
      default:
        pbSkipField(buf, state, wireType)
    }
  }

  return msg
}

export interface AgoraStreamTranscriptionExtraOptions {
  credentials: AgoraSTTCredentials
  language?: string
  abortSignal?: AbortSignal
  /** RTC token for the local user (required when App Certificate is enabled) */
  token?: string
  /** Fixed channel name (required when using a token generated from Agora Console) */
  channelName?: string
  /** UID for the local user in the RTC channel */
  localUid?: string
  /** RTC token for the STT subscriber bot */
  subBotToken?: string
  /** RTC token for the STT publisher bot */
  pubBotToken?: string
  /** UID for the STT subscriber bot */
  subBotUid?: string
  /** UID for the STT publisher bot */
  pubBotUid?: string
}

export type AgoraConnectionState = 'idle' | 'connecting' | 'publishing' | 'waiting-for-bot' | 'ready' | 'disconnected'

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
  token?: string
  channelName?: string
  localUid?: string
  subBotToken?: string
  pubBotToken?: string
  subBotUid?: string
  pubBotUid?: string
  onStateChange?: (state: AgoraConnectionState) => void
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
    const botUid = options.subBotUid || '9001'
    const token = options.token || null
    const botToken = options.subBotToken || options.pubBotToken || undefined
    // Use fixed channel name when token is provided (tokens are bound to channel names)
    const channelName = options.channelName || (token ? 'airi-stt' : `airi-stt-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`)

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
        options.onStateChange?.('disconnected')
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

    // Debug: track remote users joining/leaving
    client.on('user-joined', (user) => {
      console.info('Agora STT: remote user joined:', user.uid)
      if (String(user.uid) === botUid) {
        options.onStateChange?.('ready')
      }
    })
    client.on('user-left', (user, reason) => {
      console.info('Agora STT: remote user left:', user.uid, reason)
    })
    client.on('user-published', (user, mediaType) => {
      console.info('Agora STT: remote user published:', user.uid, mediaType)
    })

    // Listen for data messages from STT publisher bot
    client.on('stream-message', (_uid: number, data: Uint8Array) => {
      try {
        const msg = parseAgoraSTTData(data)
        if (!msg || !msg.text)
          return

        if (msg.is_final) {
          const transcript = msg.text.trim()
          if (transcript) {
            text += `${transcript} `
            const delta: StreamTranscriptionDelta = {
              type: 'transcript.text.delta',
              delta: transcript,
            }
            fullStreamCtrl?.enqueue(delta)
            textStreamCtrl?.enqueue(transcript)
            console.info('Agora STT (final):', transcript)
          }
        }
        else {
          console.info('Agora STT (interim):', msg.text)
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
    options.onStateChange?.('connecting')
    console.info('Agora STT: joining channel', channelName, 'as UID', localUid, 'with token:', token ? 'yes' : 'no')
    await client.join(credentials.appId, channelName, token, Number(localUid))
    console.info('Agora STT: joined channel successfully')

    // Create and publish microphone track
    micTrack = await AgoraRTC.createMicrophoneAudioTrack({
      encoderConfig: 'speech_standard',
    })
    await client.publish([micTrack])
    options.onStateChange?.('publishing')
    console.info('Agora STT: microphone track published')

    // Start STT agent via REST API
    const joinResponse = await agoraSTTJoin(credentials, {
      name: `airi-stt-${Date.now()}`,
      languages: [language],
      maxIdleTime: 60,
      rtcConfig: {
        channelName,
        subBotUid: botUid,
        pubBotUid: botUid,
        subBotToken: botToken,
        pubBotToken: botToken,
        subscribeAudioUids: [localUid],
      },
    })

    agentId = joinResponse.agent_id
    options.onStateChange?.('waiting-for-bot')
    console.info('Agora STT: agent started', agentId, 'status:', joinResponse.status)
  }

  doStream().catch((err) => {
    console.error('Agora STT stream error:', err)
    options.onStateChange?.('disconnected')
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
