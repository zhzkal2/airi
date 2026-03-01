/**
 * Agora Real-Time Transcription (RTT) provider
 *
 * Uses Agora RTC channel + STT REST API for real-time speech-to-text.
 * Requires: App ID, Customer ID, Customer Secret from Agora Console.
 */

import type { TranscriptionProviderWithExtraOptions } from '@xsai-ext/providers/utils'

import type { AgoraStreamTranscriptionExtraOptions } from './stream-transcription'

export type { AgoraStreamTranscriptionExtraOptions } from './stream-transcription'
export { streamAgoraTranscription } from './stream-transcription'

export function createAgoraRTTProvider(
  appId: string,
  customerId: string,
  customerSecret: string,
  options?: { language?: string, token?: string, channelName?: string, botToken?: string },
): TranscriptionProviderWithExtraOptions<string, AgoraStreamTranscriptionExtraOptions> {
  return {
    transcription: (model: string, extraOptions?: AgoraStreamTranscriptionExtraOptions) => {
      return {
        baseURL: 'https://api.agora.io',
        model: model || 'agora-rtt-v1',
        fetch: async () => {
          throw new Error('Agora RTT does not support direct fetch-based transcription. Use the streaming API.')
        },
        credentials: {
          appId,
          customerId,
          customerSecret,
        },
        language: extraOptions?.language || options?.language || 'en-US',
        token: extraOptions?.token || options?.token || undefined,
        channelName: extraOptions?.channelName || options?.channelName || undefined,
        abortSignal: extraOptions?.abortSignal,
        localUid: extraOptions?.localUid,
        subBotToken: extraOptions?.subBotToken || options?.botToken || undefined,
        pubBotToken: extraOptions?.pubBotToken || options?.botToken || undefined,
        subBotUid: extraOptions?.subBotUid,
        pubBotUid: extraOptions?.pubBotUid,
        onStateChange: extraOptions?.onStateChange,
      } as any
    },
  }
}
