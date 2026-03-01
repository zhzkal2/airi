<script setup lang="ts">
import type { HearingTranscriptionResult } from '@proj-airi/stage-ui/stores/modules/hearing'
import type { TranscriptionProviderWithExtraOptions } from '@xsai-ext/providers/utils'

import vadWorkletUrl from '@proj-airi/stage-ui/workers/vad/process.worklet?worker&url'

import {
  Alert,
  ProviderBasicSettings,
  ProviderSettingsContainer,
  ProviderSettingsLayout,
} from '@proj-airi/stage-ui/components'
import { useProviderValidation } from '@proj-airi/stage-ui/composables/use-provider-validation'
import { useHearingStore } from '@proj-airi/stage-ui/stores/modules/hearing'
import { useProvidersStore } from '@proj-airi/stage-ui/stores/providers'
import { Button, FieldInput } from '@proj-airi/ui'
import { storeToRefs } from 'pinia'
import { computed, onBeforeUnmount, reactive, ref, shallowRef } from 'vue'

const providerId = 'agora-rtt-transcription'
const defaultModel = 'agora-rtt-v1'
const SAMPLE_RATE = 16000

const languageOptions = [
  { label: 'English (US)', value: 'en-US' },
  { label: 'Japanese', value: 'ja-JP' },
  { label: 'Chinese (Simplified)', value: 'zh-CN' },
  { label: 'Chinese (Traditional)', value: 'zh-TW' },
  { label: 'Korean', value: 'ko-KR' },
  { label: 'Spanish', value: 'es-ES' },
  { label: 'French', value: 'fr-FR' },
  { label: 'German', value: 'de-DE' },
  { label: 'Portuguese (Brazil)', value: 'pt-BR' },
  { label: 'Italian', value: 'it-IT' },
]

const MAX_LANGUAGES = 2

const hearingStore = useHearingStore()
const providersStore = useProvidersStore()
const { providers } = storeToRefs(providersStore) as { providers: import('@vueuse/core').RemovableRef<Record<string, any>> }

providersStore.initializeProvider(providerId)

const credentials = reactive({
  get appId() {
    return providers.value[providerId]?.appId || ''
  },
  set appId(value: string) {
    ensureProviderCredentials()
    providers.value[providerId].appId = value
  },
  get customerId() {
    return providers.value[providerId]?.customerId || ''
  },
  set customerId(value: string) {
    ensureProviderCredentials()
    providers.value[providerId].customerId = value
  },
  get customerSecret() {
    return providers.value[providerId]?.customerSecret || ''
  },
  set customerSecret(value: string) {
    ensureProviderCredentials()
    providers.value[providerId].customerSecret = value
  },
  get languages() {
    const val = providers.value[providerId]?.languages
    return Array.isArray(val) ? val : ['en-US']
  },
  set languages(value: string[]) {
    ensureProviderCredentials()
    providers.value[providerId].languages = value
  },
  get languages() {
    return providers.value[providerId]?.languages || [] as string[]
  },
  set languages(value: string[]) {
    ensureProviderCredentials()
    providers.value[providerId].languages = value
  },
  get token() {
    return providers.value[providerId]?.token || ''
  },
  set token(value: string) {
    ensureProviderCredentials()
    providers.value[providerId].token = value
  },
  get channelName() {
    return providers.value[providerId]?.channelName || ''
  },
  set channelName(value: string) {
    ensureProviderCredentials()
    providers.value[providerId].channelName = value
  },
  get botToken() {
    return providers.value[providerId]?.botToken || ''
  },
  set botToken(value: string) {
    ensureProviderCredentials()
    providers.value[providerId].botToken = value
  },
})


function ensureProviderCredentials() {
  if (!providers.value[providerId]) {
    providers.value[providerId] = {
      appId: '',
      customerId: '',
      customerSecret: '',
      languages: ['en-US'],
      token: '',
      channelName: '',
      botToken: '',
    }
  }
}

const credentialsReady = computed(() => {
  return Boolean(
    credentials.appId.trim()
    && credentials.customerId.trim()
    && credentials.customerSecret.trim(),
  )
})

const isRecording = ref(false)
const isStreaming = ref(false)
const errorMessage = ref<string | null>(null)
const transcripts = ref<Array<{ text: string, final: boolean }>>([])

const audioContext = shallowRef<AudioContext>()
const workletNode = shallowRef<AudioWorkletNode>()
const mediaStream = shallowRef<MediaStream>()
const mediaStreamSource = shallowRef<MediaStreamAudioSourceNode>()
const audioStreamController = shallowRef<ReadableStreamDefaultController<ArrayBuffer>>()
const transcriptionAbortController = shallowRef<AbortController>()
const activeTranscription = shallowRef<HearingTranscriptionResult | null>(null)
const transcriptionTextPromise = shallowRef<Promise<string> | null>(null)

const canStart = computed(() => credentialsReady.value && !isRecording.value && !isStreaming.value)
const canStop = computed(() => isRecording.value || isStreaming.value)
const canAbort = computed(() => isStreaming.value && Boolean(transcriptionAbortController.value))

const {
  t,
  router,
  providerMetadata,
  isValidating,
  isValid,
  validationMessage,
  handleResetSettings,
  forceValid,
} = useProviderValidation(providerId)

function float32ToInt16(buffer: Float32Array) {
  const output = new Int16Array(buffer.length)
  for (let i = 0; i < buffer.length; i++) {
    const value = Math.max(-1, Math.min(1, buffer[i]))
    output[i] = value < 0 ? value * 0x8000 : value * 0x7FFF
  }
  return output
}

async function initializeAudioGraph(stream: MediaStream) {
  const context = new AudioContext({
    sampleRate: SAMPLE_RATE,
    latencyHint: 'interactive',
  })
  await context.audioWorklet.addModule(vadWorkletUrl)

  const node = new AudioWorkletNode(context, 'vad-audio-worklet-processor')
  node.port.onmessage = ({ data }: MessageEvent<{ buffer?: Float32Array }>) => {
    const buffer = data.buffer
    const controller = audioStreamController.value
    if (!buffer || !controller)
      return

    const pcm16 = float32ToInt16(buffer)
    controller.enqueue(pcm16.buffer.slice(0))
  }

  const source = context.createMediaStreamSource(stream)
  source.connect(node)

  const silentGain = context.createGain()
  silentGain.gain.value = 0
  node.connect(silentGain)
  silentGain.connect(context.destination)

  audioContext.value = context
  workletNode.value = node
  mediaStreamSource.value = source
}

function resetTranscriptionOutput() {
  transcripts.value = []
}

async function startStreaming() {
  if (!canStart.value)
    return

  errorMessage.value = null
  resetTranscriptionOutput()

  const abortController = new AbortController()
  transcriptionAbortController.value = abortController

  const audioStream = new ReadableStream<ArrayBuffer>({
    start(controller) {
      audioStreamController.value = controller
    },
    cancel: () => {
      audioStreamController.value = undefined
    },
  })

  try {
    // Dispose cached instance so fresh credentials (including token) are picked up
    await providersStore.disposeProviderInstance(providerId)
    const provider = await providersStore.getProviderInstance<TranscriptionProviderWithExtraOptions<string, any>>(providerId)
    if (!provider)
      throw new Error('Failed to initialize Agora RTT provider.')

    const result = await hearingStore.transcription(
      providerId,
      provider,
      defaultModel,
      { inputAudioStream: audioStream },
      undefined,
      {
        providerOptions: {
          abortSignal: abortController.signal,
        },
      },
    )

    if (result.mode !== 'stream')
      throw new Error('Agora RTT returned a non-streaming result unexpectedly.')

    activeTranscription.value = result
    transcriptionTextPromise.value = result.text
      .catch((error) => {
        errorMessage.value = error instanceof Error ? error.message : String(error)
        throw error
      })

    // Read text stream for real-time display
    if (result.textStream) {
      void (async () => {
        try {
          const reader = result.textStream.getReader()
          while (true) {
            const { done, value } = await reader.read()
            if (done)
              break
            if (value) {
              transcripts.value.push({ text: value, final: true })
            }
          }
        }
        catch (err) {
          console.error('Error reading Agora STT text stream:', err)
        }
      })()
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: SAMPLE_RATE,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    })

    mediaStream.value = stream
    await initializeAudioGraph(stream)

    if (audioContext.value?.state === 'suspended')
      await audioContext.value.resume()

    isRecording.value = true
    isStreaming.value = true
  }
  catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error)
    await stopStreaming()
  }
}

async function stopStreaming() {
  try {
    workletNode.value?.port.postMessage({ type: 'stop' })
  }
  catch { /* noop */ }

  if (mediaStreamSource.value) {
    mediaStreamSource.value.disconnect()
    mediaStreamSource.value = undefined
  }

  if (workletNode.value) {
    workletNode.value.port.onmessage = null
    workletNode.value.disconnect()
    workletNode.value = undefined
  }

  if (mediaStream.value) {
    mediaStream.value.getTracks().forEach(track => track.stop())
    mediaStream.value = undefined
  }

  if (audioContext.value) {
    try {
      await audioContext.value.close()
    }
    catch { /* noop */ }
    audioContext.value = undefined
  }

  audioStreamController.value?.close()
  audioStreamController.value = undefined

  isRecording.value = false

  if (transcriptionTextPromise.value) {
    try {
      await transcriptionTextPromise.value
    }
    catch { /* handled in promise */ }
    finally {
      transcriptionTextPromise.value = null
    }
  }

  isStreaming.value = false
  transcriptionAbortController.value = undefined
  activeTranscription.value = null
}

function abortStreaming() {
  const controller = transcriptionAbortController.value
  if (!controller)
    return

  controller.abort(new DOMException('Aborted by user', 'AbortError'))
  audioStreamController.value?.error(new DOMException('Aborted by user', 'AbortError'))
  audioStreamController.value = undefined
  void stopStreaming()
}

onBeforeUnmount(async () => {
  abortStreaming()
  await stopStreaming()
})
</script>

<template>
  <ProviderSettingsLayout
    :provider-name="providerMetadata?.localizedName"
    :provider-icon="providerMetadata?.icon"
    :provider-icon-color="providerMetadata?.iconColor"
    :on-back="() => router.back()"
  >
    <div class="flex flex-col gap-6 md:flex-row">
      <ProviderSettingsContainer class="w-full md:w-[40%] space-y-6">
        <ProviderBasicSettings
          :title="t('settings.pages.providers.common.section.basic.title')"
          :description="t('settings.pages.providers.common.section.basic.description')"
          :on-reset="handleResetSettings"
        >
          <FieldInput
            v-model="credentials.appId"
            label="App ID"
            placeholder="Enter your Agora App ID"
          />

          <FieldInput
            v-model="credentials.customerId"
            label="Customer ID"
            placeholder="Enter your Agora Customer ID"
          />

          <FieldInput
            v-model="credentials.customerSecret"
            label="Customer Secret"
            type="password"
            placeholder="****************"
          />

          <div flex="~ col gap-2">
            <div text-sm font-medium>
              Languages <span text="xs neutral-400">(up to {{ MAX_LANGUAGES }})</span>
            </div>
            <div flex="~ col gap-1.5">
              <label
                v-for="opt in languageOptions"
                :key="opt.value"
                flex="~ row gap-2" cursor-pointer items-center
              >
                <input
                  type="checkbox"
                  :value="opt.value"
                  :checked="credentials.languages.includes(opt.value)"
                  :disabled="!credentials.languages.includes(opt.value) && credentials.languages.length >= MAX_LANGUAGES"
                  @change="(e) => {
                    const checked = (e.target as HTMLInputElement).checked
                    if (checked && credentials.languages.length < MAX_LANGUAGES) {
                      credentials.languages = [...credentials.languages, opt.value]
                    }
                    else if (!checked) {
                      credentials.languages = credentials.languages.filter(l => l !== opt.value)
                    }
                  }"
                >
                <span text-sm :class="!credentials.languages.includes(opt.value) && credentials.languages.length >= MAX_LANGUAGES ? 'text-neutral-400' : ''">
                  {{ opt.label }}
                </span>
              </label>
            </div>
            <div v-if="credentials.languages.length === 0" text="xs red-500">
              At least one language is required
            </div>
          </div>

          <FieldInput
            v-model="credentials.token"
            label="RTC Token (optional)"
            type="password"
            placeholder="Temporary token from Agora Console"
          />

          <FieldInput
            v-model="credentials.channelName"
            label="Channel Name (optional)"
            placeholder="airi-stt (must match the token's channel)"
          />

          <FieldInput
            v-model="credentials.botToken"
            label="Bot RTC Token (optional)"
            type="password"
            placeholder="Token for STT bot UID 9001"
          />
        </ProviderBasicSettings>

        <Alert v-if="!isValid && isValidating === 0 && validationMessage" type="error">
          <template #title>
            <div class="w-full flex items-center justify-between">
              <span>{{ t('settings.dialogs.onboarding.validationFailed') }}</span>
              <button
                type="button"
                class="ml-2 rounded bg-red-100 px-2 py-0.5 text-xs text-red-600 font-medium transition-colors dark:bg-red-800/30 hover:bg-red-200 dark:text-red-300 dark:hover:bg-red-700/40"
                @click="forceValid"
              >
                {{ t('settings.pages.providers.common.continueAnyway') }}
              </button>
            </div>
          </template>
          <template #content>
            <div class="whitespace-pre-wrap break-all">
              {{ validationMessage }}
            </div>
          </template>
        </Alert>

        <Alert v-if="isValid && isValidating === 0" type="success">
          <template #title>
            {{ t('settings.dialogs.onboarding.validationSuccess') }}
          </template>
        </Alert>
      </ProviderSettingsContainer>

      <div class="w-full flex flex-1 flex-col gap-6">
        <div class="border border-neutral-200/80 rounded-xl bg-neutral-50/60 p-4 dark:border-neutral-700 dark:bg-neutral-900/40">
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div class="space-x-3">
              <Button :disabled="!canStart" variant="primary" @click="startStreaming">
                {{ isRecording ? 'Streaming...' : 'Start Realtime Transcription' }}
              </Button>
              <Button :disabled="!canStop" variant="secondary" @click="stopStreaming">
                Stop
              </Button>
              <Button
                v-if="isStreaming"
                :disabled="!canAbort"
                @click="abortStreaming"
              >
                Abort Session
              </Button>
            </div>
            <div class="text-sm text-neutral-500 dark:text-neutral-400">
              <span v-if="isRecording" class="rounded bg-red-500/10 px-2 py-0.5 text-xs text-red-500">
                Recording
              </span>
              <span v-else-if="isStreaming" class="rounded bg-blue-500/10 px-2 py-0.5 text-xs text-blue-500">
                Connected
              </span>
            </div>
          </div>

          <p v-if="errorMessage" class="mt-3 text-sm text-red-500">
            {{ errorMessage }}
          </p>
        </div>

        <div class="border border-neutral-200/80 rounded-xl bg-neutral-50/60 p-4 dark:border-neutral-700 dark:bg-neutral-900/40">
          <h2 class="text-lg font-semibold">
            Transcripts
          </h2>
          <div v-if="!transcripts.length" class="mt-3 text-sm text-neutral-400 dark:text-neutral-600">
            Waiting for audio...
          </div>
          <ul class="mt-4 text-sm space-y-3">
            <li
              v-for="(sentence, index) in transcripts"
              :key="index"
              class="flex items-start gap-3"
            >
              <span class="mt-0.5 rounded bg-neutral-200/80 px-2 py-0.5 text-xs text-neutral-700 dark:bg-neutral-800/70 dark:text-neutral-200">
                #{{ index + 1 }}
              </span>
              <div class="font-medium">
                {{ sentence.text }}
              </div>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </ProviderSettingsLayout>
</template>

<route lang="yaml">
meta:
  layout: settings
  stageTransition:
    name: slide
</route>
